/**
 * The insights dashboard.
 *
 * The behaviour worth pinning: reviews are one per line, the headline is the
 * mixed-sentence count, an empty aspect list is explained rather than shown
 * as a failure, and the page states that the aspects did not come from the
 * model. The API is mocked -- this is about the page, not the backend.
 */

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import InsightsPage from './InsightsPage'
import { ApiRequestError, getInsights } from '../services/api'
import type { InsightsResponse } from '../types'

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api')
  return { ...actual, getInsights: vi.fn() }
})

const mockedGetInsights = vi.mocked(getInsights)

const RESULT: InsightsResponse = {
  reviews: 2,
  sentences: 3,
  predictions: 4,
  mixed_sentences: 2,
  mixed_aspects: 1,
  truncated: false,
  model_version: 'absa-joint-v1',
  aspects: [
    {
      aspect: 'screen',
      source: 'corpus',
      total: 2,
      counts: { negative: 1, positive: 1 },
      shares: { negative: 0.5, positive: 0.5 },
      dominant: 'negative',
      mean_confidence: 0.98,
      mixed: true,
      examples: [
        {
          sentence: 'Battery life is amazing but the screen is dim.',
          surface: 'screen',
          sentiment: 'negative',
          confidence: 0.99,
          review_index: 0,
        },
      ],
    },
  ],
}

const EMPTY: InsightsResponse = {
  ...RESULT,
  reviews: 1,
  sentences: 1,
  predictions: 0,
  mixed_sentences: 0,
  mixed_aspects: 0,
  aspects: [],
}

beforeEach(() => {
  mockedGetInsights.mockReset()
})

function setup(domainKey: string | null = 'laptop') {
  const user = userEvent.setup()
  render(<InsightsPage domain={null} domainKey={domainKey} />)
  return { user }
}

async function typeReviews(user: ReturnType<typeof userEvent.setup>, text: string) {
  await user.click(screen.getByRole('textbox'))
  await user.paste(text)
}

/**
 * The hero figure, scoped to its own card.
 *
 * `getByText('2')` is ambiguous on this fixture -- the mixed count, the
 * review count and the aspect totals are all small integers -- so the
 * headline is read from the element beside its label.
 */
function heroValue(): HTMLElement {
  const label = screen.getByText(/sentences with two aspects and two different verdicts/i)
  const value = label.nextElementSibling
  if (!value) throw new Error('no hero value rendered')
  return value as HTMLElement
}

/** The textarea, typed. */
function textarea(): HTMLTextAreaElement {
  return screen.getByRole('textbox') as HTMLTextAreaElement
}

/** Wait for the results to render; "screen" appears in several places. */
async function waitForAspectRow(name: string) {
  await waitFor(() => expect(screen.getAllByText(name).length).toBeGreaterThan(0))
}

describe('submitting a review set', () => {
  it('cannot be analyzed while the box is empty', () => {
    setup()
    expect(screen.getByRole('button', { name: /analyze by aspect/i })).toBeDisabled()
  })

  it('sends one review per non-blank line', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'First review.\n\n  \nSecond review.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await waitFor(() => expect(mockedGetInsights).toHaveBeenCalled())
    const [payload] = mockedGetInsights.mock.calls[0]
    expect(payload.reviews).toEqual(['First review.', 'Second review.'])
  })

  it('counts the reviews as they are typed', async () => {
    const { user } = setup()
    await typeReviews(user, 'One.\nTwo.\nThree.')
    expect(screen.getByText('3 reviews')).toBeInTheDocument()
  })

  it('passes the served domain so extraction is narrowed', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup('restaurant')
    await typeReviews(user, 'Great food.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))
    await waitFor(() => expect(mockedGetInsights).toHaveBeenCalled())
    expect(mockedGetInsights.mock.calls[0][0].domain).toBe('restaurant')
  })

  it('sends no domain for a pooled model, which serves both', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup(null)
    await typeReviews(user, 'Great food.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))
    await waitFor(() => expect(mockedGetInsights).toHaveBeenCalled())
    expect(mockedGetInsights.mock.calls[0][0].domain).toBeUndefined()
  })

  it('says it is working while the request is in flight', async () => {
    mockedGetInsights.mockReturnValue(new Promise(() => {}))
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/one forward pass per pair/i)
  })
})

describe('the headline', () => {
  it('leads with the mixed-sentence count', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await waitFor(() =>
      expect(
        screen.getByText(/sentences with two aspects and two different verdicts/i),
      ).toBeInTheDocument(),
    )
    expect(heroValue()).toHaveTextContent('2')
    expect(screen.getByText(/wrong about at least one aspect/i)).toBeInTheDocument()
  })

  it('says so plainly when no sentence is mixed, rather than claiming a win', async () => {
    mockedGetInsights.mockResolvedValue({ ...RESULT, mixed_sentences: 0 })
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    expect(
      await screen.findByText(/sentence-level sentiment would do nearly as well/i),
    ).toBeInTheDocument()
  })

  it('reports the counts and the serving model', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await screen.findByText('absa-joint-v1')
    expect(screen.getByText('Reviews')).toBeInTheDocument()
    expect(screen.getByText('Aspect judgements')).toBeInTheDocument()
  })
})

describe('the results', () => {
  it('renders a row per aspect with a table view behind it', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await waitForAspectRow('screen')
    expect(screen.getByText('Show as a table')).toBeInTheDocument()
    // The legend is present because the bars carry three series. Scoped to
    // list items: "neutral" is also a column header in the table view.
    const legendEntries = screen
      .getAllByRole('listitem')
      .map((item) => item.textContent)
    expect(legendEntries).toContain('neutral')
  })

  it('explains an empty result instead of showing it as a failure', async () => {
    mockedGetInsights.mockResolvedValue(EMPTY)
    const { user } = setup()
    await typeReviews(user, 'Pollination rates among the hives were excellent.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    expect(await screen.findByText(/no aspects found in those reviews/i)).toBeInTheDocument()
    expect(screen.getByText(/the model judges aspects, it does not find them/i))
      .toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('states that the aspects did not come from the model', async () => {
    // The limitation the paste asked to be upfront about.
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await waitForAspectRow('screen')
    expect(screen.getByText(/has no aspect extraction step/i)).toBeInTheDocument()
  })

  it('warns when the backend sampled rather than judged everything', async () => {
    mockedGetInsights.mockResolvedValue({ ...RESULT, truncated: true })
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    expect(await screen.findByText(/the counts are not totals/i)).toBeInTheDocument()
  })

  it('says nothing about sampling when nothing was sampled', async () => {
    mockedGetInsights.mockResolvedValue(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await waitForAspectRow('screen')
    expect(screen.queryByText(/the counts are not totals/i)).not.toBeInTheDocument()
  })
})

describe('failure', () => {
  it('surfaces the API’s own message', async () => {
    mockedGetInsights.mockRejectedValue(new ApiRequestError('model not loaded', 503))
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('model not loaded')
  })

  it('falls back to generic copy for an unrecognisable failure', async () => {
    mockedGetInsights.mockRejectedValue('something odd')
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /could not analyze those reviews/i,
    )
  })

  it('clears a previous result rather than showing it beside an error', async () => {
    mockedGetInsights.mockResolvedValueOnce(RESULT)
    const { user } = setup()
    await typeReviews(user, 'The screen is dim.')
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))
    await waitForAspectRow('screen')

    mockedGetInsights.mockRejectedValueOnce(new ApiRequestError('gone', 503))
    await user.click(screen.getByRole('button', { name: /analyze by aspect/i }))

    await screen.findByRole('alert')
    expect(screen.queryByText('Show as a table')).not.toBeInTheDocument()
  })
})

describe('the example set', () => {
  it('fills the box for a known domain', async () => {
    const { user } = setup('laptop')
    await user.click(screen.getByRole('button', { name: /load an example set/i }))
    expect(textarea().value).toContain('Battery life is amazing but the screen is dim.')
  })

  it('is not offered when the served corpus is unknown', () => {
    setup(null)
    expect(
      screen.queryByRole('button', { name: /load an example set/i }),
    ).not.toBeInTheDocument()
  })

  it('can be cleared again', async () => {
    const { user } = setup('laptop')
    await user.click(screen.getByRole('button', { name: /load an example set/i }))
    await user.click(screen.getByRole('button', { name: /^clear$/i }))
    expect(textarea().value).toBe('')
  })
})
