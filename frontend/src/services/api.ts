/** Typed client for the ABSA backend. */

import axios, { type AxiosInstance } from 'axios'

import type {
  ApiError,
  HealthResponse,
  ModelMetaResponse,
  PredictRequest,
  PredictResponse,
} from '../types'

const API_PREFIX = '/api/v1'

/** Where a runtime override of the API origin is remembered between visits. */
const OVERRIDE_KEY = 'absa.apiUrl'

/** True for a string that parses as an http(s) URL. */
function isHttpUrl(value: string): boolean {
  try {
    const { protocol } = new URL(value)
    return protocol === 'http:' || protocol === 'https:'
  } catch {
    return false
  }
}

/**
 * A valid API origin supplied at runtime, or `null`.
 *
 * `?api=<url>` wins and is remembered, so a reload or an in-app navigation
 * keeps working once the parameter is gone; `?api=` with no value forgets it.
 * Anything that is not an http(s) URL is ignored rather than thrown, because
 * a typo in a pasted URL should leave the build-time default in place instead
 * of breaking the page.
 */
function readOverride(): string | null {
  if (typeof window === 'undefined') return null

  let candidate: string | null = null
  try {
    const fromQuery = new URLSearchParams(window.location.search).get('api')
    if (fromQuery === null) {
      candidate = window.localStorage.getItem(OVERRIDE_KEY)
    } else {
      candidate = fromQuery.trim()
      if (candidate) window.localStorage.setItem(OVERRIDE_KEY, candidate)
      else window.localStorage.removeItem(OVERRIDE_KEY)
    }
  } catch {
    // Storage is unavailable in some private-browsing modes. Reading the
    // query parameter still works, so fall back to it rather than give up.
    try {
      candidate = new URLSearchParams(window.location.search).get('api')
    } catch {
      return null
    }
  }

  if (!candidate || !isHttpUrl(candidate)) return null
  return candidate.replace(/\/+$/, '')
}

/**
 * The API's origin.
 *
 * Resolution order: a runtime override (see {@link readOverride}), then the
 * value compiled in by Vite, then the port `uvicorn app.main:app` serves on.
 *
 * The override exists because the deployed frontend points at a backend run
 * from a laptop through an ephemeral tunnel, whose hostname changes on every
 * restart. `VITE_API_URL` is substituted at *build* time, so without a
 * runtime path the site would have to be rebuilt and redeployed before each
 * demo. Note the trade-off: anyone can aim the deployed page at a host of
 * their choosing via `?api=`, so treat it as a demo affordance, not a
 * trusted setting.
 */
export const BASE_URL: string =
  readOverride() ?? import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * A failed request, thrown by every function in this module.
 *
 * A class rather than a bare object literal so it is a real `Error` (with a
 * stack and a usable message in the console) while still satisfying the
 * {@link ApiError} shape. Use {@link isApiError} to narrow in a `catch`, where
 * TypeScript types the binding as `unknown`.
 */
export class ApiRequestError extends Error implements ApiError {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
  }
}

/** Type guard for use in `catch` blocks. */
export function isApiError(error: unknown): error is ApiError {
  return (
    error instanceof ApiRequestError ||
    (typeof error === 'object' &&
      error !== null &&
      typeof (error as ApiError).message === 'string' &&
      typeof (error as ApiError).status === 'number')
  )
}

/** The backend's error body: `{"error": "...", "status": 422}`. */
interface BackendErrorBody {
  error?: unknown
  detail?: unknown
}

/**
 * Pull a human-readable message out of whatever the server sent.
 *
 * The backend answers `{error, status}` on every failure, but a proxy or a
 * crash upstream can return HTML or nothing at all, so each step is guarded.
 */
function messageFromBody(body: unknown, fallback: string): string {
  if (typeof body === 'string' && body.trim() !== '') return body
  if (typeof body === 'object' && body !== null) {
    const { error, detail } = body as BackendErrorBody
    if (typeof error === 'string' && error !== '') return error
    if (typeof detail === 'string' && detail !== '') return detail
  }
  return fallback
}

/** Convert any thrown value into an {@link ApiRequestError}. */
function toApiError(error: unknown): ApiRequestError {
  if (axios.isAxiosError(error)) {
    const { response } = error
    if (response) {
      return new ApiRequestError(
        messageFromBody(response.data, `Request failed with status ${response.status}`),
        response.status,
      )
    }
    // No response at all: the server is down, or CORS blocked it. Status 0
    // distinguishes that from any real HTTP status the server could return.
    return new ApiRequestError(
      `Could not reach the API at ${BASE_URL}. Is the backend running?`,
      0,
    )
  }
  if (error instanceof Error) return new ApiRequestError(error.message, 0)
  return new ApiRequestError('An unexpected error occurred.', 0)
}

const client: AxiosInstance = axios.create({
  baseURL: `${BASE_URL}${API_PREFIX}`,
  timeout: 20_000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Classify the sentiment expressed about `aspect` within `text`.
 *
 * @throws {ApiRequestError} on any non-2xx response or transport failure.
 */
export async function predictSentiment(
  text: string,
  aspect: string,
): Promise<PredictResponse> {
  const payload: PredictRequest = { text, aspect }
  try {
    const { data } = await client.post<PredictResponse>('/predict', payload)
    return data
  } catch (error) {
    throw toApiError(error)
  }
}

/**
 * Fetch the served model's metadata and evaluation metrics.
 *
 * @param signal - optional abort signal, so a component can cancel on unmount.
 * @throws {ApiRequestError} on any non-2xx response or transport failure.
 */
export async function getModelMeta(signal?: AbortSignal): Promise<ModelMetaResponse> {
  try {
    const { data } = await client.get<ModelMetaResponse>('/model', { signal })
    return data
  } catch (error) {
    throw toApiError(error)
  }
}

/**
 * Check whether the service is up and the model is loaded.
 *
 * @throws {ApiRequestError} on any non-2xx response or transport failure.
 */
export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const { data } = await client.get<HealthResponse>('/health', { signal })
    return data
  } catch (error) {
    throw toApiError(error)
  }
}
