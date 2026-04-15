export interface ToolCall {
  tool: string
  input: Record<string, unknown>
  result: string
}

export interface AgentSession {
  session_id: string
  task: string
  status: 'running' | 'completed' | 'failed'
  messages: unknown[]
  tool_calls: ToolCall[]
  result: string | null
  created_at: string
  completed_at: string | null
}
