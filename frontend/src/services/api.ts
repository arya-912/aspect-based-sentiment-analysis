/** Typed client for the ABSA backend. */

import axios, { type AxiosInstance } from 'axios'

import type {
  ApiError,
  HealthResponse,
  ModelMetaResponse,
  PredictRequest,
  PredictResponse,
} from '../types'

/** Falls back to the port `uvicorn app.main:app` serves on by default. */
export const BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const API_PREFIX = '/api/v1'

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
