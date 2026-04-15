import { useState, useEffect, useCallback } from 'react'
import { getSession } from '../api'
import type { AgentSession } from '../types'

const POLL_MS = 2000

export function useSession(id: string | null) {
  const [session, setSession] = useState<AgentSession | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchOnce = useCallback(async () => {
    if (!id) return null
    try {
      return await getSession(id)
    } catch {
      return null
    }
  }, [id])

  useEffect(() => {
    if (!id) return

    let cancelled = false

    setLoading(true)
    fetchOnce().then(data => {
      if (!cancelled) {
        setSession(data)
        setLoading(false)
      }
    })

    const interval = setInterval(async () => {
      const data = await getSession(id).catch(() => null)
      if (!cancelled && data) {
        setSession(data)
        if (data.status !== 'running') clearInterval(interval)
      }
    }, POLL_MS)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [id, fetchOnce])

  // Reset session when id becomes null
  const derivedSession = id ? session : null

  return { session: derivedSession, loading }
}
