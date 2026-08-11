"""One-command local demo: API + RAG + UI, with preflight checks.

Brings up `app.main:app` on :8000 and the Vite dev server on :5173, waits for
both to answer, and opens the browser. Ctrl+C stops both.

Not a deployment path - this is the reproducible local cut described in
DEPLOYMENT.md, wrapped so a first-time runner does not have to know that the
Vite proxy hardcodes :8000 or that DB_PATH has to be overridden on Windows.

Usage:
  python scripts/demo.py              # start everything
  python scripts/demo.py --setup      # fetch corpus + build the index, then start
  python scripts/demo.py --check      # preflight only, start nothing
  python scripts/demo.py --no-ui      # API only (curl / probe sweeps)
"""
import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = platform.system() == "Windows"

API_PORT = 8000  # not configurable: frontend/vite.config.ts proxies to it
UI_PORT = 5173

# Dense is the default retriever, so the demo is not meaningful without it.
# The other two only matter once the routing agent picks semantic / pdr.
RAW_DIR = ROOT / "data" / "raw"
REQUIRED_INDEX = ROOT / "data" / "chroma_middle_earth"
OPTIONAL_INDICES = [ROOT / "data" / "chroma_semantic", ROOT / "data" / "chroma_pdr"]

# The scraped corpus is published as a release asset rather than committed:
# it is 2,296 verbatim wiki articles, so keeping it out of git history leaves
# it separately versionable and easy to withdraw. Override with CORPUS_URL.
CORPUS_TAG = "corpus-v1"
CORPUS_ASSET = "corpus.zip"
DEFAULT_REPO = "nvojvodic-white/agent-platform"

BUILD_HINT = """
  The Chroma indices are gitignored (~520 MB), so a fresh clone has none.

  If you were given a data/raw archive, unzip it into data/ and run only the
  embed step - about 5 minutes and ~$0.04 of OpenAI embeddings:

      {py} -m app.rag.ingestion.build_index

  Otherwise scrape the wikis first. That is the slow part: 2,296 articles at a
  1 s/request delay, so ~40 minutes, and it depends on Tolkien Gateway staying
  reachable (it supplies 70% of the corpus and has returned 403 before):

      {py} -m app.rag.ingestion.fetch
      {py} -m app.rag.ingestion.build_index

  Optional extra indices, only needed for the semantic / pdr routes
  (~$0.12 more, a few minutes each):

      {py} -m app.rag.ingestion.build_semantic_index
      {py} -m app.rag.ingestion.build_pdr_index

  A prebuilt data/ directory copied from another machine also works - the
  indices are portable across OSes.
"""


def venv_python() -> Path:
    return ROOT / "venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def corpus_url() -> str:
    """Release-asset URL, derived from the git remote so forks work."""
    if os.environ.get("CORPUS_URL"):
        return os.environ["CORPUS_URL"]
    repo = DEFAULT_REPO
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if "github.com" in remote:
            path = remote.split("github.com")[-1].lstrip(":/")
            repo = path[:-4] if path.endswith(".git") else path
    except (OSError, subprocess.SubprocessError):
        pass
    return f"https://github.com/{repo}/releases/download/{CORPUS_TAG}/{CORPUS_ASSET}"


def rel(path: Path) -> str:
    """Repo-relative form, so printed commands are copy-pasteable."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_env_file() -> dict:
    """Parse .env without importing dotenv, so --check works pre-install."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    out = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def port_owner(port: int) -> str:
    """Best-effort 'pid 1234 (python.exe)' for whatever holds the port.

    A previous run that was killed at the wrapper rather than with Ctrl+C
    leaves its children behind, so this is the common case in practice - name
    the offender instead of making the user go hunting.
    """
    try:
        if IS_WINDOWS:
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
                    pid = parts[4]
                    name = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                        capture_output=True, text=True, timeout=15,
                    ).stdout.split(",")[0].strip('" \n')
                    return f"pid {pid} ({name})"
        else:
            pid = subprocess.run(
                ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=15,
            ).stdout.split()
            if pid:
                return f"pid {pid[0]}"
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return "unknown process"


def kill_hint(port: int) -> str:
    return (
        f'taskkill /F /PID <pid>   (find it: netstat -ano | findstr :{port})'
        if IS_WINDOWS
        else f"kill $(lsof -ti:{port} -sTCP:LISTEN)"
    )


def wait_for(url: str, timeout: int, proc: subprocess.Popen) -> bool:
    """Poll url until it answers, giving up if the process dies first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def download_corpus() -> bool:
    """Fetch and unpack the raw-article archive into data/raw."""
    url = corpus_url()
    dest = ROOT / "data" / CORPUS_ASSET
    dest.parent.mkdir(exist_ok=True)
    print(f"  downloading corpus from {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        print(f"  download failed ({e.code}). The release may not be published yet.")
        print(f"  You can scrape instead:  {rel(venv_python())} -m app.rag.ingestion.fetch")
        return False
    except (urllib.error.URLError, OSError) as e:
        print(f"  download failed: {e}")
        return False

    print(f"  unpacking {dest.stat().st_size / 1e6:.1f} MB ...")
    try:
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
            # Tolerate both layouts: an archive rooted at raw/, or one rooted at
            # the per-source directories themselves.
            rooted_at_raw = any(n.split("/")[0] == "raw" for n in names if n.strip())
            z.extractall(ROOT / "data" if rooted_at_raw else RAW_DIR)
    except zipfile.BadZipFile:
        print("  archive is not a valid zip.")
        return False
    finally:
        dest.unlink(missing_ok=True)

    count = len(list(RAW_DIR.rglob("*.json")))
    if count == 0:
        print(f"  no articles found under {rel(RAW_DIR)} after unpacking.")
        return False
    print(f"  corpus ready: {count} articles in {rel(RAW_DIR)}")
    return True


def setup(assume_yes: bool) -> bool:
    """Get from a clean clone to a usable dense index."""
    py = venv_python()

    if not RAW_DIR.exists() or not any(RAW_DIR.rglob("*.json")):
        print("\n  No corpus found. It can be downloaded (~4 MB) instead of scraped.")
        if not confirm("  Download the corpus archive?", assume_yes):
            return False
        if not download_corpus():
            return False
    else:
        print(f"  corpus already present ({rel(RAW_DIR)})")

    if REQUIRED_INDEX.exists():
        print("  dense index already built")
        return True

    print(
        "\n  Building the dense index embeds ~2.1M tokens with OpenAI\n"
        "  text-embedding-3-small: roughly 5 minutes and about $0.04."
    )
    if not confirm("  Build it now?", assume_yes):
        return False

    print("  embedding (progress below)...\n")
    rc = subprocess.run(
        [str(py), "-m", "app.rag.ingestion.build_index"], cwd=str(ROOT)
    ).returncode
    if rc != 0:
        print("\n  build_index failed - check the OpenAI key and the output above.")
        return False
    print(f"\n  index built at {rel(REQUIRED_INDEX)}")
    return True


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{question} yes (--yes)")
        return True
    if not sys.stdin.isatty():
        print(f"{question} no - not a terminal. Re-run with --yes to proceed.")
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def preflight(want_ui: bool) -> list:
    """Return [(code, message)] for each blocking problem; empty means good.

    Codes let the caller reason about *which* check failed - a missing index is
    recoverable via setup(), a busy port is not.
    """
    problems = []
    py = venv_python()

    if not py.exists():
        problems.append((
            "venv",
            f"No venv at {rel(py)}.\n"
            f"  Create it:  uv venv venv --python 3.11\n"
            f"              uv pip install --python {rel(py)} -r requirements.txt",
        ))
    else:
        probe = subprocess.run(
            [str(py), "-c", "import fastapi, anthropic, chromadb, langchain"],
            capture_output=True,
        )
        if probe.returncode != 0:
            problems.append((
                "deps",
                f"venv exists but dependencies are missing.\n"
                f"  Fix:  uv pip install --python {rel(py)} -r requirements.txt",
            ))

    env = read_env_file()
    if not env:
        problems.append(
            ("env", "No .env. Copy .env.example to .env and fill in your keys.")
        )
    else:
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            value = env.get(key, "")
            if not value or value.startswith("your_"):
                problems.append(
                    ("env", f"{key} is not set in .env (needed for the RAG path).")
                )

    if not REQUIRED_INDEX.exists():
        problems.append((
            "index",
            f"Missing dense index at {rel(REQUIRED_INDEX)}."
            + BUILD_HINT.format(py=rel(py)),
        ))

    for port in [API_PORT] + ([UI_PORT] if want_ui else []):
        if port_in_use(port):
            problems.append((
                "port",
                f"Port {port} is in use by {port_owner(port)}.\n"
                f"  Free it:  {kill_hint(port)}",
            ))

    if want_ui:
        if not shutil.which("node"):
            problems.append(
                ("node", "node not found. Install Node 20.19+ (or 22.12+).")
            )
        else:
            raw = subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            ).stdout.strip()
            major = int(raw.lstrip("v").split(".")[0])
            if major < 20:
                problems.append((
                    "node",
                    f"Node {raw} is too old - Vite needs 20.19+ or 22.12+. "
                    f"Use --no-ui to run the API alone.",
                ))

    return problems


def warnings() -> list:
    out = []
    missing = [p.name for p in OPTIONAL_INDICES if not p.exists()]
    if missing:
        out.append(
            f"Optional indices absent ({', '.join(missing)}): the routing agent "
            f"still answers, but routes that select them return nothing."
        )
    return out


def spawn(cmd: list, cwd: Path, env: dict) -> subprocess.Popen:
    """Start a child in its own process group so we can kill its whole tree."""
    kwargs = {"cwd": str(cwd), "env": env}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def stop(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    print(f"  stopping {label}...")
    if IS_WINDOWS:
        # npm spawns node as a child; terminate() would orphan it and leave the
        # port held, so kill the whole tree by pid.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local demo stack.")
    parser.add_argument("--check", action="store_true", help="preflight only")
    parser.add_argument("--setup", action="store_true",
                        help="fetch the corpus and build the index, then start")
    parser.add_argument("--yes", action="store_true",
                        help="answer yes to setup prompts (non-interactive)")
    parser.add_argument("--no-ui", action="store_true", help="API only, no frontend")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    args = parser.parse_args()
    want_ui = not args.no_ui

    print("agent-platform demo\n" + "=" * 60)

    problems = preflight(want_ui)

    # A missing index is recoverable, so offer to build it whenever setup's own
    # prerequisites hold. Unrelated failures (a busy port) still block the
    # start, but should not stop the slow, expensive step from being done now.
    setup_blockers = {"venv", "deps", "env"}
    wants_setup = args.setup or any(code == "index" for code, _ in problems)
    if wants_setup and not args.check:
        blocked = [m for code, m in problems if code in setup_blockers]
        if blocked:
            print("\nCannot set up yet:\n")
            for m in blocked:
                print(f"  - {m}\n")
            return 1
        if setup(args.yes):
            problems = preflight(want_ui)

    if problems:
        print("\nCannot start:\n")
        for _, m in problems:
            print(f"  - {m}\n")
        return 1

    for w in warnings():
        print(f"  note: {w}\n")
    print("  preflight OK")

    if args.check:
        return 0

    py = venv_python()
    env = os.environ.copy()
    # .env supplies the API keys, but DB_PATH is usually absent from it and the
    # "/data/sessions.db" default is not writable outside the container.
    env.setdefault("DB_PATH", str(ROOT / "data" / "sessions.db"))
    (ROOT / "data").mkdir(exist_ok=True)

    procs = []
    try:
        print(f"  starting API on :{API_PORT} ...")
        api = spawn(
            [str(py), "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(API_PORT)],
            cwd=ROOT,
            env=env,
        )
        procs.append((api, "API"))
        if not wait_for(f"http://127.0.0.1:{API_PORT}/health", 60, api):
            print("  API failed to start - see the traceback above.")
            return 1
        print(f"  API   ready  ->  http://localhost:{API_PORT}")

        if want_ui:
            print(f"  starting UI on :{UI_PORT} ...")
            npm = "npm.cmd" if IS_WINDOWS else "npm"
            frontend = ROOT / "frontend"
            if not (frontend / "node_modules").exists():
                print("  installing frontend dependencies (first run only)...")
                subprocess.run([npm, "install"], cwd=str(frontend), check=True)
            ui = spawn([npm, "run", "dev"], cwd=frontend, env=env)
            procs.append((ui, "UI"))
            if not wait_for(f"http://localhost:{UI_PORT}/", 90, ui):
                print("  UI failed to start.")
                return 1
            print(f"  UI    ready  ->  http://localhost:{UI_PORT}")

        target = f"http://localhost:{UI_PORT if want_ui else API_PORT}"
        print("\n" + "=" * 60)
        print(f"  Open:  {target}")
        print(f"  Docs:  http://localhost:{API_PORT}/docs")
        print("\n  Try from another terminal:")
        print(f'    curl -s -X POST http://localhost:{API_PORT}/api/v1/rag/agent_query_debug \\')
        print('      -H "content-type: application/json" \\')
        print('      -d \'{"question":"Who killed Smaug?"}\'')
        print("\n  Ctrl+C to stop.")
        print("=" * 60 + "\n")

        if not args.no_browser:
            webbrowser.open(target)

        while all(p.poll() is None for p, _ in procs):
            time.sleep(0.5)
        for p, label in procs:
            if p.poll() is not None:
                print(f"\n  {label} exited unexpectedly (code {p.returncode}).")
        return 1
    except KeyboardInterrupt:
        print("\n\nshutting down...")
        return 0
    finally:
        for p, label in reversed(procs):
            stop(p, label)
        print("  stopped.")


if __name__ == "__main__":
    sys.exit(main())
