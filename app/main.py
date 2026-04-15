from fastapi import FastAPI, Response
from app.api.routes import router
from app.observability.tracing import setup_tracing
from app.observability.metrics import get_metrics
from app.middleware.auth import APIKeyMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

setup_tracing()

app = FastAPI(
    title="Agent Platform",
    description="Self-hosted platform for running and observing Claude-powered AI agents",
    version="0.1.0"
)

app.add_middleware(APIKeyMiddleware)
FastAPIInstrumentor.instrument_app(app)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)
