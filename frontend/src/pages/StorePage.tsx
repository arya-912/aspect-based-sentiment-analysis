import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import AspectInsightRow from '../components/AspectInsightRow'
import AspectShareBar from '../components/AspectShareBar'
import AspectTable from '../components/AspectTable'
import AspectWeightPicker from '../components/AspectWeightPicker'
import Card from '../components/Card'
import CatalogueUpload from '../components/CatalogueUpload'
import ProsCons from '../components/ProsCons'
import SentimentLegend from '../components/SentimentLegend'
import StarRating from '../components/StarRating'
import { ChevronIcon, OfflineIcon } from '../components/icons'
import { asPercent } from '../lib/format'
import { weightedScore } from '../lib/weights'
import {
  getCatalogueInsights,
  getCatalogueStatus,
  getProduct,
  getProducts,
  isApiError,
} from '../services/api'
import type {
  CatalogueInsights,
  CatalogueStatus,
  ProductAspectScore,
  ProductDetail,
  ProductSummary,
} from '../types'

/** The shelf's default order: the catalogue as the backend lists it. */
const DEFAULT_SORT = ''

/** How often to re-ask for progress while the background pass runs. */
const POLL_MS = 1500

function scoresFor(
  slug: string,
  insights: CatalogueInsights | null,
): ProductAspectScore[] {
  return insights?.products[slug] ?? []
}

function scoreFor(
  slug: string,
  aspect: string,
  insights: CatalogueInsights | null,
): ProductAspectScore | null {
  return scoresFor(slug, insights).find((score) => score.aspect === aspect) ?? null
}

/**
 * A storefront whose shelf can be ranked by what the reviews say per aspect.
 *
 * Three things an overall star rating cannot do, which is the point of the
 * page: rank by one aspect, rank by a *weighted mix* of aspects the shopper
 * chooses, and say in a line what a product is praised and criticised for.
 *
 * It runs on the built-in demo catalogue or on an uploaded CSV, and the
 * honesty note at the bottom states which -- a page showing someone's own
 * data under the demo's caveats would be lying about its own numbers.
 *
 * **Ranking uses the Wilson lower bound, not the raw positive share.** By
 * raw share a restaurant with 3 positive sentences out of 3 ties for first
 * with one at 9 out of 10; the lower bound puts the better-evidenced one
 * ahead. Every score is printed with its `n` so the evidence is visible
 * rather than implied.
 *
 * Scoring is a background pass on the server (~2.5s per product on CPU), so
 * the shelf renders first and the scores arrive after; this polls progress
 * and says how far along it is instead of blocking.
 */
export default function StorePage() {
  const [products, setProducts] = useState<ProductSummary[] | null>(null)
  const [insights, setInsights] = useState<CatalogueInsights | null>(null)
  const [status, setStatus] = useState<CatalogueStatus | null>(null)
  const [sortAspect, setSortAspect] = useState(DEFAULT_SORT)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [category, setCategory] = useState<string>('all')
  const [openSlug, setOpenSlug] = useState<string | null>(null)
  const [detail, setDetail] = useState<ProductDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const generation = useRef(0)

  /** Fetch the shelf, the status and whatever scores exist. */
  const load = useCallback(async () => {
    const mine = ++generation.current
    try {
      const [shelf, state] = await Promise.all([getProducts(), getCatalogueStatus()])
      if (mine !== generation.current) return
      setProducts(shelf)
      setStatus(state)
      setError(null)
    } catch (caught: unknown) {
      if (mine !== generation.current) return
      setError(isApiError(caught) ? caught.message : 'Could not load the catalogue.')
      return
    }

    try {
      const scored = await getCatalogueInsights()
      if (mine !== generation.current) return
      setInsights(scored)
    } catch {
      // Scores are an enhancement; the shelf stands without them.
      if (mine === generation.current) setInsights(null)
    }
  }, [])

  useEffect(() => {
    // The API is the external system this effect synchronises with, which is
    // what an effect is for. `load` sets state only after its awaits, never
    // synchronously in the effect body, so the cascading render the rule
    // warns about cannot happen here.
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  // Poll only while the server says a pass is running, and stop as soon as
  // it is not: a store left open should not keep a request loop alive.
  useEffect(() => {
    if (!insights?.scoring && !status?.scoring) return
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const [state, scored] = await Promise.all([
            getCatalogueStatus(),
            getCatalogueInsights(),
          ])
          setStatus(state)
          setInsights(scored)
        } catch {
          // A failed poll is not worth surfacing; the next one may work.
        }
      })()
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [insights, status])

  const onCatalogueChanged = useCallback(
    (next: CatalogueStatus) => {
      setStatus(next)
      setInsights(null)
      setOpenSlug(null)
      setDetail(null)
      setSortAspect(DEFAULT_SORT)
      setWeights({})
      void load()
    },
    [load],
  )

  const openProduct = useCallback(
    async (slug: string) => {
      if (openSlug === slug) {
        setOpenSlug(null)
        setDetail(null)
        return
      }
      setOpenSlug(slug)
      setDetail(null)
      try {
        setDetail(await getProduct(slug))
      } catch (caught: unknown) {
        setError(isApiError(caught) ? caught.message : 'Could not load that product.')
      }
    },
    [openSlug],
  )

  const categories = useMemo(
    () => [...new Set((products ?? []).map((product) => product.category))],
    [products],
  )

  const filtered = useMemo(() => {
    const rows = products ?? []
    return category === 'all'
      ? rows
      : rows.filter((product) => product.category === category)
  }, [products, category])

  const weightedAspects = useMemo(
    () => Object.keys(weights).filter((aspect) => (weights[aspect] ?? 0) > 0),
    [weights],
  )

  /** The shelf, in the order the current controls ask for. */
  const shelf = useMemo(() => {
    if (weightedAspects.length > 0) {
      // Weights win over the single-aspect sort: they are the more specific
      // request, and having both apply at once would be incoherent.
      return [...filtered].sort((a, b) => {
        const left = weightedScore(scoresFor(a.slug, insights), weights)
        const right = weightedScore(scoresFor(b.slug, insights), weights)
        if (!left && !right) return 0
        if (!left) return 1
        if (!right) return -1
        return right.value - left.value
      })
    }
    if (sortAspect === DEFAULT_SORT) return filtered

    // Products the aspect is not discussed on sort last rather than being
    // hidden: a shelf that silently drops items is worse than one that ranks
    // them honestly at the bottom.
    return [...filtered].sort((a, b) => {
      const left = scoreFor(a.slug, sortAspect, insights)
      const right = scoreFor(b.slug, sortAspect, insights)
      if (!left && !right) return 0
      if (!left) return 1
      if (!right) return -1
      // The Wilson lower bound, not the raw share.
      return right.score - left.score
    })
  }, [filtered, sortAspect, insights, weights, weightedAspects])

  /** Aspects valid for the current filter, so the controls offer no dead keys. */
  const sortableAspects = useMemo(() => {
    if (!insights) return []
    const slugs = new Set(filtered.map((product) => product.slug))
    const available = new Set<string>()
    for (const [slug, scores] of Object.entries(insights.products)) {
      if (!slugs.has(slug)) continue
      for (const score of scores) available.add(score.aspect)
    }
    return insights.aspects.filter((aspect) => available.has(aspect))
  }, [insights, filtered])

  const scoring = Boolean(insights?.scoring || status?.scoring)
  const scored = insights?.scored ?? status?.scored ?? 0
  const total = insights?.total ?? status?.products ?? 0
  const uploaded = status?.source === 'upload'

  if (error && !products) {
    return (
      <Card className="p-5">
        <p role="alert" className="flex items-center gap-2 text-sm text-critical">
          <OfflineIcon />
          {error}
        </p>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <h2 className="text-base font-semibold text-ink">Demo store</h2>
        <p className="mt-1 max-w-prose text-sm text-ink-2">
          Rank a shelf by what the reviews actually say about one aspect, or by a
          mix you weight yourself &mdash; neither of which an overall star
          rating can express.
        </p>

        <div className="mt-4">
          <CatalogueUpload status={status} onChanged={onCatalogueChanged} />
        </div>

        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="text-sm">
            <span className="block text-xs text-muted">Category</span>
            <select
              value={category}
              onChange={(event) => {
                setCategory(event.target.value)
                setSortAspect(DEFAULT_SORT)
              }}
              className="mt-1 rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
            >
              <option value="all">All</option>
              {categories.map((value) => (
                <option key={value} value={value}>
                  {value === 'laptop' ? 'Laptops' : 'Restaurants'}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm">
            <span className="block text-xs text-muted">Best for</span>
            <select
              value={sortAspect}
              onChange={(event) => setSortAspect(event.target.value)}
              disabled={sortableAspects.length === 0 || weightedAspects.length > 0}
              className="mt-1 rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none disabled:opacity-50"
            >
              <option value={DEFAULT_SORT}>Catalogue order</option>
              {sortableAspects.map((aspect) => (
                <option key={aspect} value={aspect}>
                  {aspect}
                </option>
              ))}
            </select>
          </label>

          {scoring && (
            <p role="status" className="pb-1.5 text-xs text-muted">
              Scoring reviews&hellip; {scored} of {total} products done.
            </p>
          )}
          {!scoring && total > 0 && scored < total && (
            <p className="pb-1.5 text-xs text-muted">
              {scored} of {total} products scored.
            </p>
          )}
        </div>

        {sortableAspects.length > 0 && (
          <div className="mt-4 border-t border-line pt-4">
            <AspectWeightPicker
              aspects={sortableAspects}
              weights={weights}
              onChange={setWeights}
            />
          </div>
        )}
      </Card>

      {(sortAspect !== DEFAULT_SORT || weightedAspects.length > 0) && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-ink-2">
            {weightedAspects.length > 0 ? (
              <>
                Ranked by your weights:{' '}
                <span className="font-medium text-ink">
                  {weightedAspects.map((aspect) => `${aspect} ×${weights[aspect]}`).join(', ')}
                </span>
              </>
            ) : (
              <>
                Ranked by <span className="font-medium text-ink">{sortAspect}</span>, on the
                lower bound of its positive share
              </>
            )}
          </p>
          <SentimentLegend />
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {shelf.map((product) => {
          const score = scoreFor(product.slug, sortAspect, insights)
          const weighted =
            weightedAspects.length > 0
              ? weightedScore(scoresFor(product.slug, insights), weights)
              : null
          const isOpen = openSlug === product.slug

          return (
            <Card key={product.slug} className="p-4">
              <div className="flex items-baseline justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate font-semibold text-ink">{product.name}</h3>
                  <p className="truncate text-xs text-muted">{product.subtitle}</p>
                </div>
                {product.price && (
                  <span className="shrink-0 text-sm text-ink-2">{product.price}</span>
                )}
              </div>

              {/* The reviewers' own score, beside the model's aspect view. */}
              {product.rating !== null && (
                <div className="mt-2">
                  <StarRating rating={product.rating} count={product.rating_count} />
                </div>
              )}

              {weighted && (
                <div className="mt-3 rounded-lg border border-line bg-plane p-2.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-xs text-ink-2">Match for your weights</span>
                    <span className="tabular text-sm font-semibold text-ink">
                      {asPercent(weighted.value, 0)}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-muted">
                    over {weighted.matched} of {weightedAspects.length} weighted{' '}
                    {weightedAspects.length === 1 ? 'aspect' : 'aspects'}
                  </p>
                </div>
              )}

              {weightedAspects.length > 0 && !weighted && (
                <p className="mt-3 text-xs text-muted">
                  None of your weighted aspects is discussed in these reviews, so this
                  cannot be compared.
                </p>
              )}

              {/* The sort key's standing, when one is chosen. */}
              {weightedAspects.length === 0 && sortAspect !== DEFAULT_SORT && (
                <div className="mt-3">
                  {score ? (
                    <>
                      <div className="flex items-baseline justify-between gap-2 text-xs">
                        <span className="text-ink-2">{sortAspect}</span>
                        <span className="tabular text-ink">
                          {asPercent(score.positive_share, 0)} positive
                          <span className="text-muted">
                            {' '}
                            of {score.total} {score.total === 1 ? 'sentence' : 'sentences'}
                          </span>
                        </span>
                      </div>
                      <div className="mt-1.5">
                        <AspectShareBar
                          aspect={sortAspect}
                          shares={{
                            positive: score.positive_share,
                            negative: score.negative_share,
                            neutral: Math.max(
                              0,
                              1 - score.positive_share - score.negative_share,
                            ),
                          }}
                          counts={{
                            positive: Math.round(score.positive_share * score.total),
                            negative: Math.round(score.negative_share * score.total),
                            neutral: Math.max(
                              0,
                              score.total -
                                Math.round(score.positive_share * score.total) -
                                Math.round(score.negative_share * score.total),
                            ),
                          }}
                        />
                      </div>
                      <p className="mt-1 text-xs text-muted">
                        Ranked on {score.score.toFixed(2)} &mdash; the lower bound of that
                        share over {score.total}{' '}
                        {score.total === 1 ? 'sentence' : 'sentences'}
                      </p>
                    </>
                  ) : (
                    <p className="text-xs text-muted">
                      Not discussed on &ldquo;{sortAspect}&rdquo; in these reviews.
                    </p>
                  )}
                </div>
              )}

              <button
                type="button"
                onClick={() => void openProduct(product.slug)}
                aria-expanded={isOpen}
                className="mt-3 flex items-center gap-1 text-sm font-medium text-accent hover:underline"
              >
                <span className={`transition-transform ${isOpen ? 'rotate-0' : '-rotate-90'}`}>
                  <ChevronIcon className="h-3.5 w-3.5" />
                </span>
                {product.review_count} reviews, by aspect
                {product.reviews_seen > product.review_count && (
                  <span className="font-normal text-muted">
                    {' '}
                    (of {product.reviews_seen})
                  </span>
                )}
              </button>

              {isOpen && (
                <div className="mt-3 border-t border-line pt-3">
                  {!detail ? (
                    <p role="status" className="text-sm text-muted">
                      Judging {product.review_count} reviews&hellip;
                    </p>
                  ) : (
                    <>
                      <ProsCons aspects={detail.insights.aspects} />

                      <div className="mt-3 flex flex-wrap items-baseline justify-between gap-2">
                        <p className="text-xs text-muted">
                          {detail.insights.sentences} sentences ·{' '}
                          {detail.insights.predictions} aspect judgements
                        </p>
                        <SentimentLegend />
                      </div>
                      <div className="mt-2">
                        {detail.insights.aspects.map((summary) => (
                          <AspectInsightRow
                            key={summary.aspect}
                            summary={summary}
                            showSource={false}
                          />
                        ))}
                      </div>
                      <details className="mt-3">
                        <summary className="cursor-pointer text-xs font-medium text-ink-2 hover:text-ink">
                          Show as a table
                        </summary>
                        <div className="mt-2">
                          <AspectTable aspects={detail.insights.aspects} />
                        </div>
                      </details>
                    </>
                  )}
                </div>
              )}
            </Card>
          )
        })}
      </div>

      {products === null && (
        <Card className="p-5">
          <p role="status" className="text-sm text-muted">
            Loading the catalogue&hellip;
          </p>
        </Card>
      )}

      {products !== null && shelf.length === 0 && (
        <Card className="p-5">
          <p className="text-sm text-ink">No products in this category.</p>
        </Card>
      )}

      {/* Which data this is. Never the demo's caveats over someone else's numbers. */}
      <div className="max-w-prose space-y-2 text-xs text-muted">
        {uploaded ? (
          <p>
            Showing <span className="font-medium text-ink-2">{status?.source_name}</span>,
            uploaded by you &mdash; not the built-in demo catalogue.{' '}
            {status?.products_found && status.products_found > (status?.products ?? 0) ? (
              <>
                The {status.products} most-reviewed of {status.products_found.toLocaleString('en-US')}{' '}
                products in the file are shown, from{' '}
                {status.rows_used.toLocaleString('en-US')} review rows.{' '}
              </>
            ) : null}
            An upload lives in the server&rsquo;s memory only, so a restart returns to the
            demo catalogue.
          </p>
        ) : (
          <p>
            Showing the built-in demo catalogue. The products are invented; the reviews
            are not &mdash; each is a real SemEval-2014 sentence selected by its gold
            polarity for the aspect in question, and no sentence appears under two
            products. Most come from the held-out test split, but the laptop split is too
            small to build contrasting products from, so some laptop reviews are
            sentences the model was trained on, which flatters it.
          </p>
        )}
        <p>
          Aspects here are fixed per-category sort keys drawn from the SemEval gold
          vocabulary, which covers laptops and restaurants only &mdash; reviews of
          anything else will show few aspects or none. Shares are over the sentences that
          mention each aspect, not over all reviews, and ranking uses the lower bound of
          the positive share so a product with three reviews cannot outrank one with
          thirty on luck alone.
        </p>
      </div>
    </div>
  )
}
