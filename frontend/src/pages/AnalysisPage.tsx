import { useState } from 'react'

import Card from '../components/Card'
import ResultCard from '../components/ResultCard'
import ReviewForm from '../components/ReviewForm'
import SentimentBadge from '../components/SentimentBadge'
import type { Example } from '../lib/domains'
import type { PredictResponse, Sentiment } from '../types'

/**
 * The predictions made so far about one review.
 *
 * Results belong to a *sentence*, not to the app: an attention overlay and a
 * mixed-polarity claim are both statements about a specific review, so they
 * have to be discarded the moment the review changes rather than left to
 * describe text that is no longer on screen. Keeping the text here is what
 * makes that check possible.
 */
interface AnalysisPageProps {
  /** Examples for the served corpus; omitted when it is unrecognised. */
  examples?: readonly Example[]
}

interface Session {
  text: string
  /** Newest first, so a fresh prediction appears where the eye already is. */
  results: PredictResponse[]
}

/**
 * Group the aspects by the verdict they got.
 *
 * Used only to decide whether this sentence disagrees with itself, and to name
 * the aspects on each side when it does.
 */
function groupBySentiment(results: PredictResponse[]): Map<Sentiment, string[]> {
  const groups = new Map<Sentiment, string[]>()
  for (const result of results) {
    const existing = groups.get(result.sentiment)
    if (existing) existing.push(result.aspect)
    else groups.set(result.sentiment, [result.aspect])
  }
  return groups
}

export default function AnalysisPage({ examples }: AnalysisPageProps) {
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(false)

  function handleResult(result: PredictResponse, text: string) {
    setSession((current) => {
      // A different sentence is a different subject; the old results described
      // text the user has since replaced.
      if (current === null || current.text !== text) return { text, results: [result] }

      // Re-analyzing the same aspect replaces its row rather than stacking a
      // duplicate, so the list stays one entry per aspect.
      const key = result.aspect.toLowerCase()
      const kept = current.results.filter((existing) => existing.aspect.toLowerCase() !== key)
      return { text, results: [result, ...kept] }
    })
  }

  const groups = session ? groupBySentiment(session.results) : new Map<Sentiment, string[]>()
  const isMixed = groups.size > 1

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      {/*
        Sticky on wide screens: the whole point of the results list is to add
        aspects to the same sentence and compare, which means the form has to
        stay reachable without scrolling back up to it.
      */}
      <div className="lg:col-span-2">
        <Card className="p-5 lg:sticky lg:top-24">
          <h2 className="text-base font-semibold text-ink">Analyze a review</h2>
          <p className="mt-1 mb-5 text-sm text-ink-2">
            Enter a sentence and an aspect term inside it. The model judges that aspect
            specifically — so the same sentence can be positive about one and negative
            about another.
          </p>
          <ReviewForm
            onResult={handleResult}
            onLoadingChange={setLoading}
            examples={examples}
          />
        </Card>
      </div>

      <div className="space-y-4 lg:col-span-3">
        {session === null ? (
          <Card className="p-8 text-center">
            <p className="text-sm font-medium text-ink">
              {loading ? 'Analyzing…' : 'No predictions yet'}
            </p>
            <p className="mx-auto mt-2 max-w-sm text-sm text-ink-2">
              {loading
                ? 'Running the review through the model.'
                : 'Pick one of the examples, or write a review and click a word in it to choose the aspect. Analyze several aspects of the same sentence to see them compared.'}
            </p>
          </Card>
        ) : (
          <>
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-xs font-medium uppercase tracking-wide text-muted">
                  {session.results.length === 1
                    ? '1 aspect analyzed'
                    : `${session.results.length} aspects analyzed`}
                </p>
                <p className="mt-1 text-sm text-ink-2">“{session.text}”</p>
              </div>
              <button
                type="button"
                onClick={() => setSession(null)}
                className="shrink-0 rounded-lg border border-line px-2.5 py-1 text-xs font-medium text-ink-2 transition-colors hover:bg-plane hover:text-ink"
              >
                Clear
              </button>
            </div>

            {/*
              The project's central claim, asserted by the data rather than by
              the prose above: this only appears when the model has actually
              returned different polarities for different aspects of one
              sentence.
            */}
            {isMixed && (
              <Card className="border-accent/40 bg-accent/5 p-4">
                <p className="text-sm font-medium text-ink">
                  One sentence, {groups.size} different verdicts
                </p>
                <div className="mt-3 space-y-2">
                  {[...groups].map(([sentiment, aspects]) => (
                    <div key={sentiment} className="flex flex-wrap items-center gap-2">
                      <SentimentBadge sentiment={sentiment} size="sm" />
                      <span className="text-sm text-ink-2">{aspects.join(', ')}</span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-xs text-muted">
                  A sentence-level sentiment model has to collapse this into one label.
                  Judging each aspect against its own evidence is what this model is
                  for — compare the highlighted words below to see which evidence it
                  used for each.
                </p>
              </Card>
            )}

            {loading && (
              <p className="text-sm text-ink-2" role="status">
                Analyzing…
              </p>
            )}

            {session.results.map((result) => (
              <ResultCard
                key={result.aspect.toLowerCase()}
                result={result}
                text={session.text}
              />
            ))}
          </>
        )}
      </div>
    </div>
  )
}
