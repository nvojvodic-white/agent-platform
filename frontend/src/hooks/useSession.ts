import { useState, useEffect, useCallback } from 'react'
import { getSession } from '../api'
import type { AgentSession } from '../types'

const POLL_MS = 2000

export function useSession(id: string | null) {
  const [session, setSession] = useState<AgentSession | null>(null)
  const [loading, setLoading] = useState(false)

  const fetch = useCallback(async () => {
    if (!id) return
    try {
      const data = await getSession(id)
      setSession(data)
    } catch {
      // session may have been deleted
    }
  }, [id])

  useEffect(() => {
    if (!id) { setSession(null); return }
    setLoading(true)
    fetch().finally(() => setLoading(false))

    // Keep polling while running
    const id_ = setInterval(async () => {
      const data = await getSession(id).catch(() => null)
      if (data) {
        setSession(data)
        if (data.status !== 'running') clearInterval(id_)
      }
    }, POLL_MS)

    return () => clearInterval(id_)
  }, [id, fetch])

  return { session, loading }
}
