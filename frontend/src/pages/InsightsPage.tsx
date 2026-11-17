import { useCallback, useRef, useState } from 'react'

import AspectInsightRow from '../components/AspectInsightRow'
import AspectTable from '../components/AspectTable'
import Card from '../components/Card'
import ReviewPaste from '../components/ReviewPaste'
import SentimentLegend from '../components/SentimentLegend'
import { WarningIcon } from '../components/icons'
import { asCount } from '../lib/format'
import { getInsights, isApiError } from '../services/api'
import type { Domain } from '../lib/domains'
import type { InsightsResponse } from '../types'

interface InsightsPageProps {
  /** The served artifact's domain, for the extraction vocabulary and the
   *  example set. Null for an unrecognised corpus: everything still works,
   *  the vocabulary is simply not narrowed. */
  domain: Domain | null
  domainKey: string | null
}

/** A worked example per domain: several reviews, deliberately disagreeing. */
const EXAMPLES: Record<string, string> = {
  laptop: [
    'Battery life is amazing but the screen is dim.',
    'The keyboard feels great, but the trackpad is unresponsive and the price is too high.',
    'Screen is washed out in sunlight, though the battery easily lasts a day.',
    'It comes with a 500GB hard drive and a six-cell battery.',
    'Gorgeous screen, terrible fan noise.',
    'The price is fair for the performance you get.',
  ].join('\n'),
  restaurant: [
    'Although the tables may be closely situated, the candle-light, food-quality and service overcompensate.',
    'I like the somosas, chai, and the chole, but the dhosas and dhal were kinda dissapointing.',
    'Great food but the service was dreadful.',
    'The atmosphere is wonderful and the prices are reasonable.',
    'And it all comes at a very reasonable price (congee, noodles, and rice dishes are no more than $3-6 each).',
    'Service was prompt, the food arrived cold.',
  ].join('\n'),
}

/** Split the textarea into reviews: one per non-blank line. */
function toReviews(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '')
}

/**
 * Per-aspect sentiment over a review set the user brings.
 *
 * The argument the page exists to make: every one of its `mixed_sentences` is
 * a sentence with two aspects and two different verdicts, so a single-label
 * classifier is necessarily wrong about one of them. That number is computed
 * from the user's own reviews, which is a far better demonstration than any
 * claim about the method.
 *
 * The honest caveat, stated on the page rather than buried: the model does not
 * find aspects. It judges (sentence, aspect) pairs, and the aspects here come
 * from matching the corpora's gold vocabulary. An aspect no SemEval annotator
 * marked is invisible to this page.
 */
export default function InsightsPage({ domain, domainKey }: InsightsPageProps) {
  const [text, setText] = useState('')
  const [result, setResult] = useState<InsightsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const example = domainKey ? EXAMPLES[domainKey] : undefined

  const analyze = useCallback(async () => {
    const reviews = toReviews(text)
    if (reviews.length === 0) return

    // A second click supersedes the first: these requests take seconds, and a
    // stale response arriving last would overwrite the newer one.
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setBusy(true)
    setError(null)
    try {
      const data = await getInsights(
        { reviews, domain: domainKey ?? undefined, max_aspects: 12 },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setResult(data)
    } catch (caught: unknown) {
      if (controller.signal.aborted) return
      setError(isApiError(caught) ? caught.message : 'Could not analyze those reviews.')
      setResult(null)
    } finally {
      if (!controller.signal.aborted) setBusy(false)
    }
  }, [text, domainKey])

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <h2 className="text-base font-semibold text-ink">Review insights</h2>
        <p className="mt-1 max-w-prose text-sm text-ink-2">
          Paste a product&rsquo;s reviews and get a verdict per aspect, with the
          sentences behind each one.{' '}
          {domain ? `Tuned for ${domain.subject.toLowerCase()}.` : null}
        </p>

        <div className="mt-4">
          <ReviewPaste
            value={text}
            onChange={setText}
            onAnalyze={analyze}
            busy={busy}
            onLoadExample={example ? () => setText(example) : undefined}
            exampleLabel="Load an example set"
          />
        </div>
      </Card>

      {error && (
        <Card className="p-4">
          <p role="alert" className="text-sm text-critical">
            {error}
          </p>
        </Card>
      )}

      {result && !busy && (
        <>
          {/*
            The headline. A hero figure rather than a chart: it is one number,
            and the whole case for aspect-based analysis is that it is not zero.
          */}
          <Card className="p-5">
            <p className="text-xs uppercase tracking-wide text-muted">
              Sentences with two aspects and two different verdicts
            </p>
            <p className="tabular mt-1 text-5xl font-semibold text-ink">
              {asCount(result.mixed_sentences)}
            </p>
            <p className="mt-2 max-w-prose text-sm text-ink-2">
              {result.mixed_sentences > 0 ? (
                <>
                  A classifier that labels whole sentences has to pick one
                  verdict for each of these, so it is wrong about at least one
                  aspect every time. That is the gap this model is for.
                </>
              ) : (
                <>
                  None in this set &mdash; every sentence here agrees with
                  itself. Sentence-level sentiment would do nearly as well on
                  these reviews; try a set with mixed opinions in single
                  sentences to see the difference.
                </>
              )}
            </p>
            <dl className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
              <div className="flex gap-1">
                <dt>Reviews</dt>
                <dd className="tabular text-ink-2">{asCount(result.reviews)}</dd>
              </div>
              <div className="flex gap-1">
                <dt>Sentences</dt>
                <dd className="tabular text-ink-2">{asCount(result.sentences)}</dd>
              </div>
              <div className="flex gap-1">
                <dt>Aspect judgements</dt>
                <dd className="tabular text-ink-2">{asCount(result.predictions)}</dd>
              </div>
              <div className="flex gap-1">
                <dt>Model</dt>
                <dd className="font-mono text-ink-2">{result.model_version}</dd>
              </div>
            </dl>
          </Card>

          <Card className="p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-sm font-semibold text-ink">By aspect</h3>
              <SentimentLegend />
            </div>

            {result.truncated && (
              <p className="mt-3 flex gap-2 rounded-lg border border-warning/40 bg-warning/10 p-2.5 text-xs text-ink-2">
                <span className="mt-px text-warning">
                  <WarningIcon />
                </span>
                That set was larger than one request judges, so each aspect was
                scored on a sample of its sentences. The shares are
                representative; the counts are not totals.
              </p>
            )}

            {result.aspects.length === 0 ? (
              <div className="mt-4 rounded-lg border border-line bg-plane p-4">
                <p className="text-sm text-ink">No aspects found in those reviews.</p>
                <p className="mt-1 max-w-prose text-sm text-ink-2">
                  Aspects are matched against the aspect terms human annotators
                  marked in SemEval-2014, so a product category the corpus never
                  covered will come back empty. This is the honest limit of the
                  pipeline rather than a failure of the model: the model judges
                  aspects, it does not find them.
                </p>
              </div>
            ) : (
              <>
                <div className="mt-3">
                  {result.aspects.map((summary) => (
                    <AspectInsightRow key={summary.aspect} summary={summary} />
                  ))}
                </div>

                <details className="mt-4">
                  <summary className="cursor-pointer text-xs font-medium text-ink-2 hover:text-ink">
                    Show as a table
                  </summary>
                  <div className="mt-3">
                    <AspectTable aspects={result.aspects} />
                  </div>
                </details>
              </>
            )}
          </Card>

          {/*
            Said on the page, not only in the README. The aspect list is the
            one part of this pipeline the model contributes nothing to, and a
            dashboard that implied otherwise would be overclaiming.
          */}
          <p className="max-w-prose text-xs text-muted">
            The model classifies a (sentence, aspect) pair; it has no aspect
            extraction step. The aspects above were found by matching the
            SemEval-2014 gold aspect terms against your text, and each row says
            which tier it came from. Anything those annotators never marked will
            not appear.
          </p>
        </>
      )}

      {busy && (
        <Card className="p-5">
          <p role="status" className="text-sm text-ink-2">
            Judging every aspect in every sentence&hellip; this is one forward
            pass per pair, so a large set takes a few seconds.
          </p>
        </Card>
      )}
    </div>
  )
}
