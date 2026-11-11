import { useEffect, useState } from 'react'

import { BASE_URL, getHealth, isApiError } from '../services/api'
import { OfflineIcon, WarningIcon } from './icons'

/**
 * `checking` is the first request, before anything is known; nothing renders
 * then, so a healthy load never flashes a banner.
 */
type Status = 'checking' | 'ok' | 'degraded' | 'unreachable'

/**
 * How long to wait before re-checking after a bad result.
 *
 * Polling continues only while the service is unhealthy and stops for good on
 * the first success, so a working app issues exactly one request. The retry
 * means starting the backend clears the banner on its own, without a reload.
 */
const RETRY_MS = 5000

/**
 * A banner shown only when the backend cannot serve predictions.
 *
 * It separates the two failures that look identical from inside the UI: the
 * API being unreachable, and the API being up with no model loaded. The
 * backend deliberately answers 200 `{"status": "degraded"}` for the second
 * rather than 503, which is what makes them distinguishable at all.
 *
 * Both states ship an icon beside the text. The two cases use reserved status
 * colors, and a status color never carries meaning on its own.
 */
export default function HealthBanner() {
  const [status, setStatus] = useState<Status>('checking')
  const [detail, setDetail] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    // Guards against a late response from a request this effect already
    // abandoned setting state after unmount.
    let cancelled = false

    const check = async (): Promise<void> => {
      try {
        const health = await getHealth(controller.signal)
        if (cancelled) return
        // `model_loaded` rather than `status`: the two always agree, and the
        // boolean is the thing this component branches on.
        if (health.model_loaded) {
          setStatus('ok')
          setDetail(null)
          return // Healthy: settled, so stop polling.
        }
        setStatus('degraded')
        setDetail(null)
      } catch (error: unknown) {
        if (cancelled || controller.signal.aborted) return
        setStatus('unreachable')
        setDetail(isApiError(error) ? error.message : null)
      }
      if (!cancelled) timer = setTimeout(() => void check(), RETRY_MS)
    }

    void check()

    return () => {
      cancelled = true
      controller.abort()
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [])

  if (status === 'checking' || status === 'ok') return null

  const unreachable = status === 'unreachable'

  return (
    <div
      role="alert"
      className={`mb-6 flex gap-3 rounded-xl border p-4 text-sm ${
        unreachable
          ? 'border-critical/40 bg-critical/10'
          : 'border-warning/40 bg-warning/10'
      }`}
    >
      <span className={`mt-0.5 ${unreachable ? 'text-critical' : 'text-warning'}`}>
        {unreachable ? <OfflineIcon /> : <WarningIcon />}
      </span>

      <div className="min-w-0">
        <p className="font-medium text-ink">
          {unreachable ? 'Cannot reach the API' : 'No model is loaded'}
        </p>
        <p className="mt-1 text-ink-2">
          {unreachable
            ? (detail ?? `No response from ${BASE_URL}.`)
            : 'The server is running, but it started without an artifact, so every prediction will fail.'}
        </p>
        <p className="mt-2 text-xs text-muted">
          {unreachable ? (
            <>
              Start it with{' '}
              <code className="rounded bg-plane px-1 py-0.5 font-mono text-ink-2">
                cd backend &amp;&amp; uvicorn app.main:app --port 8000
              </code>
            </>
          ) : (
            <>
              Train one with{' '}
              <code className="rounded bg-plane px-1 py-0.5 font-mono text-ink-2">
                cd training &amp;&amp; python train.py train
              </code>
            </>
          )}
          . Rechecking every {RETRY_MS / 1000}s.
        </p>
      </div>
    </div>
  )
}
