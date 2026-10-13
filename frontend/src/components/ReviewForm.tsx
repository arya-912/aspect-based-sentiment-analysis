import { useState } from 'react'

import { isApiError, predictSentiment } from '../services/api'
import type { PredictResponse } from '../types'

interface ReviewFormProps {
  onResult: (result: PredictResponse) => void
  /** Lets the page react while a request is in flight. Optional. */
  onLoadingChange?: (loading: boolean) => void
}

const MAX_TEXT = 2000
const MAX_ASPECT = 200

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

  function setBusy(busy: boolean) {
    setLoading(busy)
    onLoadingChange?.(busy)
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setApiError(null)

    if (isEmpty) {
      setValidationError('Both a review text and an aspect term are required.')
      return
    }
    setValidationError(null)

    setBusy(true)
    try {
      onResult(await predictSentiment(trimmedText, trimmedAspect))
    } catch (error) {
      setApiError(
        isApiError(error) ? error.message : 'Something went wrong. Please try again.',
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-4">
      <div>
        <label htmlFor="review-text" className="block text-sm font-medium text-gray-700">
          Review text
        </label>
        <textarea
          id="review-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
          maxLength={MAX_TEXT}
          placeholder="The battery life is great but the screen is dim."
          className="mt-1 w-full rounded-md border border-gray-300 p-2 text-sm focus:border-blue-500 focus:outline-none"
        />
        <p className="mt-1 text-xs text-gray-500">
          {trimmedText.length}/{MAX_TEXT} characters
        </p>
      </div>

      <div>
        <label htmlFor="aspect" className="block text-sm font-medium text-gray-700">
          Aspect term
        </label>
        <input
          id="aspect"
          type="text"
          value={aspect}
          onChange={(event) => setAspect(event.target.value)}
          maxLength={MAX_ASPECT}
          placeholder="screen"
          className="mt-1 w-full rounded-md border border-gray-300 p-2 text-sm focus:border-blue-500 focus:outline-none"
        />
        <p className="mt-1 text-xs text-gray-500">
          A word or phrase from the review — the thing to judge.
        </p>
      </div>

      {validationError !== null && (
        <p role="alert" className="text-sm text-amber-700">
          {validationError}
        </p>
      )}

      {apiError !== null && (
        <p
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800"
        >
          {apiError}
        </p>
      )}

      <button
        type="submit"
        disabled={isEmpty || loading}
        className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500"
      >
        {loading ? 'Analyzing…' : 'Analyze sentiment'}
      </button>
    </form>
  )
}
