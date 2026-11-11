import { useState } from 'react'

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
}

const MAX_TEXT = 2000
const MAX_ASPECT = 200

/**
 * Examples worth one click each, chosen to show a different thing about the
 * model rather than to look varied.
 *
 * The empty state used to be a sentence of prose telling the user to type
 * something. The interesting behaviour of this model -- one sentence, opposite
 * sentiments; a neutral that means "mentioned, not judged"; a confident answer
 * about an aspect that is not there -- is hard to stumble into from a blank box
 * and immediate from these.
 */
const EXAMPLES: ReadonlyArray<{
  label: string
  hint: string
  text: string
  aspect: string
}> = [
  {
    label: 'Opposite in one sentence',
    hint: 'Analyze “screen”, then “Battery life”, and compare',
    text: 'Battery life is amazing but the screen is dim.',
    aspect: 'screen',
  },
  {
    label: 'Three aspects',
    hint: 'One review, three verdicts',
    text: 'The keyboard feels great, but the trackpad is unresponsive and the price is too high.',
    aspect: 'trackpad',
  },
  {
    label: 'Mentioned, not judged',
    hint: 'What “neutral” actually means here',
    text: 'It comes with a 500GB hard drive and a six-cell battery.',
    aspect: 'hard drive',
  },
  {
    label: 'An aspect that is absent',
    hint: 'Confidently wrong, and flagged for it',
    text: 'The battery life is great but the screen is dim.',
    aspect: 'keyboard',
  },
]

/** Sentence + aspect input, submitting to the predict endpoint. */
export default function ReviewForm({ onResult, onLoadingChange }: ReviewFormProps) {
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

  // A hint, not a verdict. The backend decides this properly on token ids --
  // it sees through casing and possessives and knows about truncation -- so
  // this only ever says "you may have meant something else", before a request.
  const maybeAbsent = !isEmpty && !looksPresent(trimmedText, trimmedAspect)

  function setBusy(busy: boolean) {
    setLoading(busy)
    onLoadingChange?.(busy)
  }

  function applyExample(example: (typeof EXAMPLES)[number]) {
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
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Try an example
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
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
          aria-describedby={maybeAbsent ? 'aspect-absent-hint' : undefined}
        />

        {maybeAbsent && (
          <p
            id="aspect-absent-hint"
            className="mt-1.5 flex items-start gap-1.5 text-xs text-ink-2"
          >
            <span className="mt-px text-warning">
              <WarningIcon className="h-3.5 w-3.5" />
            </span>
            <span>
              “{trimmedAspect}” does not look like a phrase from this review. The model
              will still answer, confidently — which is the point worth seeing.
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
