import { useSession } from '../hooks/useSession'
import { StatusBadge } from './StatusBadge'
import type { ToolCall } from '../types'

interface Props {
  sessionId: string | null
}

function ToolCallCard({ tc }: { tc: ToolCall }) {
  return (
    <div className="tool-call">
      <div className="tool-call-name">⚙ {tc.name}</div>
      <pre className="tool-call-body">{JSON.stringify(tc.input, null, 2)}</pre>
      {tc.output !== undefined && (
        <pre className="tool-call-output">{tc.output}</pre>
      )}
    </div>
  )
}

export function SessionDetail({ sessionId }: Props) {
  const { session, loading } = useSession(sessionId)

  if (!sessionId) {
    return (
      <div className="detail-empty">
        <p>Select a session to see details</p>
      </div>
    )
  }

  if (loading && !session) {
    return <div className="detail-empty"><p>Loading…</p></div>
  }

  if (!session) {
    return <div className="detail-empty"><p>Session not found</p></div>
  }

  const durationMs = session.completed_at
    ? new Date(session.completed_at).getTime() - new Date(session.created_at).getTime()
    : null

  return (
    <div className="session-detail">
      <div className="detail-header">
        <StatusBadge status={session.status} />
        {durationMs !== null && (
          <span className="detail-duration">{(durationMs / 1000).toFixed(1)}s</span>
        )}
      </div>

      <h2 className="detail-task">{session.task}</h2>

      <p className="detail-meta">
        ID: <code>{session.session_id}</code> ·{' '}
        {new Date(session.created_at).toLocaleString()}
      </p>

      {session.tool_calls.length > 0 && (
        <section className="detail-section">
          <h3>Tool Calls ({session.tool_calls.length})</h3>
          {session.tool_calls.map((tc, i) => (
            <ToolCallCard key={i} tc={tc} />
          ))}
        </section>
      )}

      {session.result && (
        <section className="detail-section">
          <h3>Result</h3>
          <div className="result-box">{session.result}</div>
        </section>
      )}

      {session.status === 'running' && (
        <div className="running-indicator">
          <span className="pulse" /> Agent is working…
        </div>
      )}
    </div>
  )
}
