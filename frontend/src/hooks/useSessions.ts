import { useState, useEffect, useCallback } from 'react'
import { listSessions } from '../api'
import type { AgentSession } from '../types'

const POLL_MS = 3000

export function useSessions() {
  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetch = useCallback(async () => {
    try {
      const data = await listSessions()
      setSessions(data.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ))
      setError(null)
    } catch {
      setError('Cannot reach API')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetch()
    const id = setInterval(fetch, POLL_MS)
    return () => clearInterval(id)
  }, [fetch])

  return { sessions, loading, error, refresh: fetch }
}
