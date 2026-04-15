import type { AgentSession } from './types'

const BASE = '/api/v1'

// Optional API key — set VITE_API_KEY in frontend/.env.local for local dev
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined

function headers(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...extra }
  if (API_KEY) h['X-API-Key'] = API_KEY
  return h
}

export async function listSessions(): Promise<AgentSession[]> {
  const res = await fetch(`${BASE}/sessions`, { headers: headers() })
  if (!res.ok) throw new Error('Failed to fetch sessions')
  return res.json()
}

export async function getSession(id: string): Promise<AgentSession> {
  const res = await fetch(`${BASE}/sessions/${id}`, { headers: headers() })
  if (!res.ok) throw new Error('Failed to fetch session')
  return res.json()
}

export async function createSession(task: string): Promise<{ session_id: string; status: string }> {
  const res = await fetch(`${BASE}/sessions`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ task }),
  })
  if (!res.ok) throw new Error('Failed to create session')
  return res.json()
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${BASE}/sessions/${id}`, { method: 'DELETE', headers: headers() })
}
