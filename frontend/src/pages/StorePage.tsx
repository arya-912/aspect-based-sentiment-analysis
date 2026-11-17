import { useCallback, useEffect, useMemo, useState } from 'react'

import AspectInsightRow from '../components/AspectInsightRow'
import AspectShareBar from '../components/AspectShareBar'
import AspectTable from '../components/AspectTable'
import Card from '../components/Card'
import SentimentLegend from '../components/SentimentLegend'
import { ChevronIcon, OfflineIcon } from '../components/icons'
import { asPercent } from '../lib/format'
import { getCatalogueInsights, getProduct, getProducts, isApiError } from '../services/api'
import type {
  CatalogueInsights,
  ProductAspectScore,
  ProductDetail,
  ProductSummary,
} from '../types'

/** The shelf's default order: the catalogue as the backend lists it. */
const DEFAULT_SORT = ''

function scoreFor(
  slug: string,
  aspect: string,
  insights: CatalogueInsights | null,
): ProductAspectScore | null {
  if (!insights) return null
  return insights.products[slug]?.find((score) => score.aspect === aspect) ?? null
}

/**
 * A mock storefront whose shelf can be sorted by one aspect.
 *
 * This is the query no overall rating can answer. "Best battery" and "best
 * screen" are different orderings of the same products, and a single star
 * rating -- or a single sentence-level sentiment score -- collapses them into
 * one. Changing the sort here visibly reorders the shelf, which is the whole
 * demonstration.
 *
 * The products are fictional. Every review sentence is a real SemEval-2014
 * sentence, picked by its gold polarity for the aspect in question, so the
 * contrast between products is grounded in human annotation rather than in
 * text written to produce it. Most are from the held-out test split; the page
 * says so, per product, because a sentence the model trained on is not a fair
 * demonstration.
 *
 * Two requests, deliberately: the shelf renders from `/products` immediately,
 * and the aspect scores arrive separately because scoring the whole catalogue
 * is ~8s of CPU on a cold process. The sort control stays disabled and says
 * why until they land, rather than the page sitting blank.
 */
export default function StorePage() {
  const [products, setProducts] = useState<ProductSummary[] | null>(null)
  const [insights, setInsights] = useState<CatalogueInsights | null>(null)
  const [sortAspect, setSortAspect] = useState(DEFAULT_SORT)
  const [category, setCategory] = useState<string>('all')
  const [openSlug, setOpenSlug] = useState<string | null>(null)
  const [detail, setDetail] = useState<ProductDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scoring, setScoring] = useState(true)

  useEffect(() => {
    const controller = new AbortController()

    getProducts(controller.signal)
      .then(setProducts)
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        setError(isApiError(caught) ? caught.message : 'Could not load the catalogue.')
      })

    // Fired alongside, not after: the shelf does not wait for it.
    getCatalogueInsights(controller.signal)
      .then(setInsights)
      .catch(() => undefined)
      .finally(() => {
        if (!controller.signal.aborted) setScoring(false)
      })

    return () => controller.abort()
  }, [])

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

  const shelf = useMemo(() => {
    let rows = products ?? []
    if (category !== 'all') rows = rows.filter((product) => product.category === category)
    if (sortAspect === DEFAULT_SORT) return rows

    // Products the aspect is not discussed on sort last rather than being
    // hidden: a shelf that silently drops items is worse than one that ranks
    // them honestly at the bottom.
    return [...rows].sort((a, b) => {
      const left = scoreFor(a.slug, sortAspect, insights)
      const right = scoreFor(b.slug, sortAspect, insights)
      if (!left && !right) return 0
      if (!left) return 1
      if (!right) return -1
      return right.positive_share - left.positive_share
    })
  }, [products, category, sortAspect, insights])

  /** Aspects valid for the current filter, so the control offers no dead keys. */
  const sortableAspects = useMemo(() => {
    if (!insights) return []
    const slugs = new Set(shelf.map((product) => product.slug))
    const available = new Set<string>()
    for (const [slug, scores] of Object.entries(insights.products)) {
      if (!slugs.has(slug)) continue
      for (const score of scores) available.add(score.aspect)
    }
    return insights.aspects.filter((aspect) => available.has(aspect))
  }, [insights, shelf])

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
          Eight fictional products, each with real reviews. Sort the shelf by one
          aspect and the order changes &mdash; which is the thing an overall star
          rating cannot tell you.
        </p>

        {/* Filters in one row above the shelf. */}
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
              disabled={scoring || sortableAspects.length === 0}
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
              Scoring every review in the catalogue&hellip; a few seconds, once.
            </p>
          )}
        </div>
      </Card>

      {sortAspect !== DEFAULT_SORT && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-ink-2">
            Ranked by positive share for <span className="font-medium text-ink">{sortAspect}</span>
          </p>
          <SentimentLegend />
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {shelf.map((product) => {
          const score = scoreFor(product.slug, sortAspect, insights)
          const isOpen = openSlug === product.slug

          return (
            <Card key={product.slug} className="p-4">
              <div className="flex items-baseline justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate font-semibold text-ink">{product.name}</h3>
                  <p className="truncate text-xs text-muted">{product.subtitle}</p>
                </div>
                <span className="shrink-0 text-sm text-ink-2">{product.price}</span>
              </div>

              {/* The sort key's standing, when one is chosen. */}
              {sortAspect !== DEFAULT_SORT && (
                <div className="mt-3">
                  {score ? (
                    <>
                      <div className="flex items-baseline justify-between gap-2 text-xs">
                        <span className="text-ink-2">{sortAspect}</span>
                        <span className="tabular text-ink">
                          {asPercent(score.positive_share, 0)} positive
                          <span className="text-muted"> of {score.total}</span>
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
              </button>

              {isOpen && (
                <div className="mt-3 border-t border-line pt-3">
                  {!detail ? (
                    <p role="status" className="text-sm text-muted">
                      Judging {product.review_count} reviews&hellip;
                    </p>
                  ) : (
                    <>
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
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

      <p className="max-w-prose text-xs text-muted">
        The products are invented; the reviews are not. Each is a real
        SemEval-2014 sentence selected by its gold polarity for the aspect in
        question, and no sentence appears under two products. Most come from the
        held-out test split, but the laptop split is too small to build
        contrasting products from, so some laptop reviews are sentences the
        model was trained on &mdash; which flatters it. Aspects here are the
        fixed per-category sort keys, not extracted.
      </p>
    </div>
  )
}
