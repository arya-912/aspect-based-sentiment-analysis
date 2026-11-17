/**
 * The "best for me" weighting: turning per-aspect scores into one ranking.
 *
 * Lives in `lib/` rather than beside the picker because the picker is a
 * component file, and a module that exports both a component and a helper
 * breaks fast refresh.
 */

/** The weights on offer. 0 is a real choice: it drops the aspect entirely. */
export const WEIGHT_STEPS = [0, 1, 2, 3] as const

export interface WeightedResult {
  value: number
  /** How many of the weighted aspects this product is actually discussed on. */
  matched: number
}

/**
 * The weighted score for one product.
 *
 * A weighted mean of the per-aspect Wilson lower bounds, over the weighted
 * aspects the product is actually discussed on.
 *
 * Returns null when it is discussed on none of them, which the shelf renders
 * as "cannot be compared" rather than as a zero: scoring it 0 would rank an
 * unreviewed product below a thoroughly disliked one, which is not what the
 * data says.
 *
 * The denominator counts only the aspects found, so a product missing one
 * weighted aspect is not penalised for the absence -- it is ranked on what is
 * known about it, and the card prints how many aspects that was, so a product
 * matched on one of three weighted aspects cannot quietly look as
 * well-evidenced as one matched on all three.
 */
export function weightedScore(
  scores: { aspect: string; score: number }[],
  weights: Record<string, number>,
): WeightedResult | null {
  let total = 0
  let weightSum = 0
  let matched = 0

  for (const score of scores) {
    const weight = weights[score.aspect] ?? 0
    if (weight <= 0) continue
    total += score.score * weight
    weightSum += weight
    matched += 1
  }

  if (weightSum === 0) return null
  return { value: total / weightSum, matched }
}
