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

/** One review token and the share of attention it received. */
export interface TokenAttention {
  /** A whole word: the backend folds WordPiece `##` pieces back together. */
  token: string
  /** Attention weight in 0-1, summed over the word's pieces. */
  weight: number
}

/** Response of `POST /api/v1/predict`. */
export interface PredictResponse {
  aspect: string
  sentiment: Sentiment
  /** Softmax probability of the chosen label; null if the backend omits it. */
  confidence: number | null
  /**
   * Softmax probability of every label, keyed by label name. Null when the
   * backend cannot line its label names up with the logits.
   */
  scores: Record<string, number> | null
  /**
   * The most-attended review tokens, heaviest first. Null when the served
   * model reports no attention, so always check before rendering.
   */
  attention: TokenAttention[] | null
  /**
   * Whether the aspect's tokens occur verbatim in the review the model saw.
   * `false` means the prediction is off-distribution -- every training aspect
   * was a literal span of its sentence -- so the UI must say so rather than
   * present the confidence at face value. `null` when the backend could not
   * tell.
   */
  aspect_in_text: boolean | null
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
  /**
   * Counts of gold label vs predicted label: `confusion_matrix[gold][pred]`,
   * indexed in class-id order, i.e. the order of {@link SENTIMENTS}.
   *
   * The backend has always sent this -- it is in its `EVALUATION_KEYS` -- but
   * nothing read it until the Performance page grew a matrix view. Optional
   * because an artifact can ship a metrics report without one.
   */
  confusion_matrix?: number[][]
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
  /**
   * Stem of the training file the artifact was fitted on, e.g.
   * `Laptop_Train_v2`. The UI's copy and its examples follow this rather than
   * `version`, which is only a name someone chose. Null when the artifact
   * records no training path; optional because an older backend omits it.
   */
  corpus?: string | null
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

/* ---- aspect insights over a review set ------------------------------- */

/** One sentence, and the model's verdict on one aspect of it. */
export interface AspectExample {
  sentence: string
  /**
   * The aspect exactly as this sentence spells it. May differ in inflection
   * from the aspect's label -- "batteries" grouped under "battery" -- because
   * the model is asked about the text really present in the sentence.
   */
  surface: string
  sentiment: Sentiment
  confidence: number | null
  /** Index into the request's `reviews`. */
  review_index: number
}

/**
 * Where an aspect came from. Never the model: it judges aspects it is handed
 * and has no extraction step, so the UI has to be able to say who chose.
 *
 * `corpus` a SemEval gold term annotated 3+ times · `corpus-rare` annotated
 * once or twice · `given` named by the caller.
 */
export type AspectSource = 'corpus' | 'corpus-rare' | 'given'

/** What a review set says about one aspect. */
export interface AspectSummary {
  aspect: string
  source: AspectSource
  /** Sentences judged for this aspect. */
  total: number
  counts: Partial<Record<Sentiment, number>>
  shares: Partial<Record<Sentiment, number>>
  dominant: Sentiment
  mean_confidence: number
  /** True when this aspect drew more than one polarity. */
  mixed: boolean
  examples: AspectExample[]
}

/** Response of `POST /api/v1/insights`. */
export interface InsightsResponse {
  reviews: number
  sentences: number
  predictions: number
  aspects: AspectSummary[]
  /**
   * Sentences where two aspects drew different polarities. Each one is a
   * sentence a single-label classifier is necessarily wrong about, which is
   * the whole case for doing this per aspect.
   */
  mixed_sentences: number
  mixed_aspects: number
  /** True when the request hit the backend's forward-pass ceiling. */
  truncated: boolean
  model_version: string
}

/** Body of `POST /api/v1/insights`. */
export interface InsightsRequest {
  reviews: string[]
  aspects?: string[]
  domain?: string
  max_aspects?: number
}

/* ---- the demo storefront --------------------------------------------- */

export interface ProductSummary {
  slug: string
  name: string
  category: string
  subtitle: string
  price: string
  review_count: number
}

export interface ProductDetail extends ProductSummary {
  reviews: string[]
  insights: InsightsResponse
}

/** One product's standing on one aspect, for sorting the shelf. */
export interface ProductAspectScore {
  aspect: string
  positive_share: number
  negative_share: number
  total: number
}

/** Response of `GET /api/v1/products/insights`. */
export interface CatalogueInsights {
  aspects: string[]
  /** Product slug -> its score on each aspect it is discussed on. */
  products: Record<string, ProductAspectScore[]>
  model_version: string
}
