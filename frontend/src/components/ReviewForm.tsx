import { useState } from 'react'

import type { Example } from '../lib/domains'
import { looksPresent } from '../lib/words'
import { isApiError, predictSentiment } from '../services/api'
import type { PredictResponse } from '../types'
import AspectPicker from './AspectPicker'
import { OfflineIcon, WarningIcon } from './icons'

interface ReviewFormProps {
  /** Receives the prediction and the review it was made about. */
  onResult: (result: PredictResponse, text: string) => void
  /** Lets the page react while a request is in flight. Optional. */
  onLoadingChange?: (loading: boolean) => void
  /**
   * One-click examples for the corpus being served; empty when the served
   * artifact is one `domains.ts` does not recognise.
   *
   * They arrive from the caller rather than living here because they are only
   * valid for one corpus: every example's aspect has to be a literal span of
   * its own sentence, so a laptop sentence offered against a restaurant model
   * teaches the wrong thing while looking authoritative.
   */
  examples?: readonly Example[]
}

const MAX_TEXT = 2000
const MAX_ASPECT = 200

/** Sentence + aspect input, submitting to the predict endpoint. */
export default function ReviewForm({
  onResult,
  onLoadingChange,
  examples = [],
}: ReviewFormProps) {
  const [text, setText] = useState('')
  const [aspect, setAspect] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Trimmed, because whitespace-only input is what the backend rejects as 422;
  // catching it here keeps the button honest and saves a round trip.
  const trimmedText = text.trim()
  const trimmedAspect = aspect.trim()
  const isEmpty = trimmedText === '' || trimmedAspect === ''

  // The same rule the server applies, so this can say so before the request
  // instead of hinting at it. `looksPresent` matches whole words on stems, as
  // `aspect_match.py` does; anything it rejects the API answers 422 for.
  const isAbsent = !isEmpty && !looksPresent(trimmedText, trimmedAspect)

  function setBusy(busy: boolean) {
    setLoading(busy)
    onLoadingChange?.(busy)
  }

  function applyExample(example: Example) {
    setText(example.text)
    setAspect(example.aspect)
    setValidationError(null)
    setApiError(null)
  }

  async function submit() {
    setApiError(null)

    if (isEmpty) {
      setValidationError('Both a review text and an aspect term are required.')
      return
    }
    // Caught here as well as server-side: the API refuses this with a 422, and
    // a round trip to be told what was already on screen is just latency.
    if (isAbsent) {
      setValidationError(
        `“${trimmedAspect}” is not a phrase from this review, so there is nothing to judge.`,
      )
      return
    }
    setValidationError(null)

    setBusy(true)
    try {
      onResult(await predictSentiment(trimmedText, trimmedAspect), trimmedText)
    } catch (error) {
      setApiError(
        isApiError(error) ? error.message : 'Something went wrong. Please try again.',
      )
    } finally {
      setBusy(false)
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await submit()
  }

  /**
   * Cmd/Ctrl+Enter submits from inside the textarea.
   *
   * A bare Enter has to keep inserting a newline -- reviews are sentences and
   * people do paste multi-line text -- so the shortcut takes a modifier, which
   * is also what every other "submit from a textarea" in the world uses.
   */
  function handleTextKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !isEmpty && !loading) {
      event.preventDefault()
      void submit()
    }
  }

  const fieldClass =
    'mt-1.5 w-full rounded-lg border border-line bg-plane p-2.5 text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none'

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-5">
      {/*
        Hidden rather than empty when the corpus is unrecognised: a heading
        over no buttons reads as a broken fetch.
      */}
      {examples.length > 0 && (
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            Try an example
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {examples.map((example) => (
              <button
                key={example.label}
                type="button"
                onClick={() => applyExample(example)}
                title={example.hint}
                className="rounded-full border border-line bg-plane px-3 py-1 text-xs font-medium text-ink-2 transition-colors hover:border-accent/50 hover:bg-accent/10 hover:text-ink"
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div>
        <label htmlFor="review-text" className="block text-sm font-medium text-ink">
          Review text
        </label>
        <textarea
          id="review-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleTextKeyDown}
          rows={3}
          maxLength={MAX_TEXT}
          placeholder="The battery life is great but the screen is dim."
          className={`${fieldClass} resize-y`}
        />
        <p className="tabular mt-1 text-right text-xs text-muted">
          {trimmedText.length}/{MAX_TEXT}
        </p>
      </div>

      <div>
        <label htmlFor="aspect" className="block text-sm font-medium text-ink">
          Aspect term
        </label>
        <input
          id="aspect"
          type="text"
          value={aspect}
          onChange={(event) => setAspect(event.target.value)}
          maxLength={MAX_ASPECT}
          placeholder="screen"
          className={fieldClass}
          aria-describedby={isAbsent ? 'aspect-absent-hint' : undefined}
        />

        {isAbsent && (
          <p
            id="aspect-absent-hint"
            className="mt-1.5 flex items-start gap-1.5 text-xs text-ink-2"
          >
            <span className="mt-px text-warning">
              <WarningIcon className="h-3.5 w-3.5" />
            </span>
            <span>
              “{trimmedAspect}” is not a phrase from this review. The model has no way
              to answer “not mentioned” — it would return a confident polarity anyway —
              so this is refused rather than guessed. Pick a word below.
            </span>
          </p>
        )}

        {trimmedText !== '' && (
          <div className="mt-3">
            <AspectPicker text={trimmedText} onPick={setAspect} selected={aspect} />
          </div>
        )}
      </div>

      {validationError !== null && (
        <p role="alert" className="text-sm text-ink-2">
          {validationError}
        </p>
      )}

      {apiError !== null && (
        <div
          role="alert"
          className="flex gap-2.5 rounded-lg border border-critical/40 bg-critical/10 p-3 text-sm text-ink"
        >
          <span className="mt-0.5 text-critical">
            <OfflineIcon />
          </span>
          <span>{apiError}</span>
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={isEmpty || loading}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-on-accent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-muted/30 disabled:text-muted"
        >
          {loading ? 'Analyzing…' : 'Analyze sentiment'}
        </button>
        <span className="hidden text-xs text-muted sm:inline">
          or <kbd className="font-mono">⌘</kbd>/<kbd className="font-mono">Ctrl</kbd> +{' '}
          <kbd className="font-mono">Enter</kbd>
        </span>
      </div>
    </form>
  )
}
