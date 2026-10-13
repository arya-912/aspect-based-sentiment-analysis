/** Shared types mirroring the backend's request and response schemas. */

/** The three polarities the model predicts. */
export type Sentiment = 'positive' | 'neutral' | 'negative'

/** All sentiments, in the model's class-id order (negative=0, neutral=1, positive=2). */
export const SENTIMENTS: readonly Sentiment[] = ['negative', 'neutral', 'positive']

/** Body of `POST /api/v1/predict`. */
export interface PredictRequest {
  text: string
  aspect: string
}

/** Response of `POST /api/v1/predict`. */
export interface PredictResponse {
  aspect: string
  sentiment: Sentiment
  /** Softmax probability of the chosen label; null if the backend omits it. */
  confidence: number | null
  model_version: string
}

/** Precision/recall/F1 for one label. */
export interface PerClassMetrics {
  f1: number
  precision: number
  recall: number
  /** Number of gold examples; sent by the backend, absent on hand-written data. */
  support?: number
}

/** The `evaluation` block of `GET /api/v1/model`. */
export interface ModelEvaluation {
  accuracy: number
  macro_f1: number
  per_class: Record<string, PerClassMetrics>
  /** Extras the backend also sends; optional so a leaner payload still fits. */
  weighted_f1?: number
  n?: number
  evaluation_split?: string
}

/** Response of `GET /api/v1/model`. */
export interface ModelMetaResponse {
  version: string
  base_model: string
  task: string
  labels: string[]
  /** Null when the artifact ships without a metrics report. */
  evaluation: ModelEvaluation | null
}

/** Response of `GET /api/v1/health`. */
export interface HealthResponse {
  status: 'ok' | 'degraded'
  model_loaded: boolean
}

/** A failed API call, normalised into one shape. */
export interface ApiError {
  message: string
  /** HTTP status, or 0 when the request never reached the server. */
  status: number
}
