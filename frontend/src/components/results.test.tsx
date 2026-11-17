/**
 * The result and metrics components: `ResultCard`, `SentimentBadge`,
 * `HighlightedSentence`, `PerClassTable`, `ConfusionMatrix`.
 *
 * What these assert is mostly about *honesty of presentation* -- the caveat
 * that has to appear above the confidence, the label that cannot be dropped in
 * favour of a colour, the row order that must not follow JSON key order.
 */

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ConfusionMatrix from './ConfusionMatrix'
import HighlightedSentence from './HighlightedSentence'
import PerClassTable from './PerClassTable'
import ResultCard from './ResultCard'
import SentimentBadge from './SentimentBadge'
import type { PredictResponse } from '../types'

const REVIEW = 'Battery life is amazing but the screen is dim.'

const RESULT: PredictResponse = {
  aspect: 'screen',
  sentiment: 'negative',
  confidence: 0.7228,
  scores: { negative: 0.7228, neutral: 0.2149, positive: 0.0623 },
  attention: [
    { token: 'dim', weight: 0.285 },
    { token: 'is', weight: 0.16 },
    { token: 'amazing', weight: 0.04 },
  ],
  aspect_in_text: true,
  model_version: 'absa-v1',
}

describe('SentimentBadge', () => {
  it.each(['negative', 'neutral', 'positive'] as const)(
    'always spells out %s, never colour alone',
    (sentiment) => {
      // The poles sit at CVD deltaE 7.2, which is legal only with a secondary
      // encoding. The text is that encoding, so it is mandatory.
      render(<SentimentBadge sentiment={sentiment} />)
      expect(screen.getByText(sentiment)).toBeInTheDocument()
    },
  )

  it('hides the decorative dot from assistive tech', () => {
    const { container } = render(<SentimentBadge sentiment="positive" />)
    expect(container.querySelector('[aria-hidden="true"]')).toBeInTheDocument()
  })
})

describe('ResultCard', () => {
  /**
   * The confidence figure, scoped to its own <dd>.
   *
   * `getByText('72.3%')` is ambiguous on purpose-built data: the same number
   * is also the negative bar's value in the distribution below. Scoping to
   * the definition list is what makes the assertion about the headline
   * confidence rather than whichever node happened to match first.
   */
  function confidenceValue(): HTMLElement {
    const term = screen.getByText('Confidence')
    const value = term.parentElement?.querySelector('dd')
    if (!value) throw new Error('no confidence value rendered')
    return value as HTMLElement
  }

  /**
   * The headline sentiment badge.
   *
   * Found by its dot rather than by text: the predicted label also appears as
   * a row label in the distribution below, so a bare text query is ambiguous
   * on any result whose label is in `scores` -- which is all of them.
   */
  function sentimentBadge(): HTMLElement {
    const badge = screen
      .getAllByText(RESULT.sentiment)
      .find((node) => node.querySelector('[aria-hidden="true"]') !== null)
    if (!badge) throw new Error('no sentiment badge rendered')
    return badge
  }

  it('shows the aspect, the label and the confidence', () => {
    render(<ResultCard result={RESULT} text={REVIEW} />)
    // By title: the word "screen" also occurs in the review text below.
    expect(screen.getByTitle('screen')).toHaveTextContent('screen')
    expect(sentimentBadge()).toHaveTextContent('negative')
    expect(confidenceValue()).toHaveTextContent('72.3%')
    expect(screen.getByText('absa-v1')).toBeInTheDocument()
  })

  it('renders an em dash rather than NaN% for a missing confidence', () => {
    render(<ResultCard result={{ ...RESULT, confidence: null }} text={REVIEW} />)
    expect(confidenceValue()).toHaveTextContent('—')
  })

  it('lists the label distribution in class-id order, not by value', () => {
    // Sorting by value would reshuffle the rows between predictions, which
    // makes two results impossible to compare at a glance.
    render(<ResultCard result={RESULT} text={REVIEW} />)
    const labels = screen
      .getAllByText(/^(negative|neutral|positive)$/)
      .map((node) => node.textContent)
    expect(labels).toEqual(['negative', 'negative', 'neutral', 'positive'])
  })

  it('omits the distribution when the backend sent no scores', () => {
    render(<ResultCard result={{ ...RESULT, scores: null }} text={REVIEW} />)
    expect(screen.queryByText('Label distribution')).not.toBeInTheDocument()
  })

  it('omits the attention section when the model reports none', () => {
    render(<ResultCard result={{ ...RESULT, attention: null }} text={REVIEW} />)
    expect(screen.queryByText('What the model looked at')).not.toBeInTheDocument()
  })

  it('omits it for an empty attention list too', () => {
    render(<ResultCard result={{ ...RESULT, attention: [] }} text={REVIEW} />)
    expect(screen.queryByText('What the model looked at')).not.toBeInTheDocument()
  })

  describe('the absent-aspect caveat', () => {
    it('warns when aspect_in_text is explicitly false', () => {
      render(
        <ResultCard
          result={{ ...RESULT, aspect: 'keyboard', aspect_in_text: false }}
          text={REVIEW}
        />,
      )
      const note = screen.getByRole('note')
      expect(note).toHaveTextContent('does not appear in this review')
      expect(note).toHaveTextContent('not trustworthy')
    })

    it('puts the caveat above the confidence, so it is read first', () => {
      render(
        <ResultCard
          result={{ ...RESULT, aspect: 'keyboard', aspect_in_text: false }}
          text={REVIEW}
        />,
      )
      const note = screen.getByRole('note')
      // The model is just as confident about an absent aspect as a present
      // one, so the order on screen is load-bearing.
      expect(note.compareDocumentPosition(confidenceValue())).toBe(
        Node.DOCUMENT_POSITION_FOLLOWING,
      )
    })

    it('says nothing when aspect_in_text is true', () => {
      render(<ResultCard result={RESULT} text={REVIEW} />)
      expect(screen.queryByRole('note')).not.toBeInTheDocument()
    })

    it('says nothing when aspect_in_text is null', () => {
      // `null` is "could not tell". A caveat built from unknown would cry wolf.
      render(<ResultCard result={{ ...RESULT, aspect_in_text: null }} text={REVIEW} />)
      expect(screen.queryByRole('note')).not.toBeInTheDocument()
    })
  })
})

describe('HighlightedSentence', () => {
  it('renders the review verbatim, punctuation and all', () => {
    const { container } = render(
      <HighlightedSentence text={REVIEW} attention={RESULT.attention ?? []} />,
    )
    expect(container.textContent).toBe(REVIEW)
  })

  it('states each highlighted word’s own weight in its title', () => {
    render(<HighlightedSentence text={REVIEW} attention={RESULT.attention ?? []} />)
    expect(screen.getByTitle(/^dim: 28\.5%/)).toBeInTheDocument()
  })

  it('shows both occurrences of a repeated word at the higher weight', () => {
    // The API reports per word, not per position: two entries for "is" arrive
    // and nothing says which occurrence each belongs to.
    render(
      <HighlightedSentence
        text="is it fine or is it not"
        attention={[
          { token: 'is', weight: 0.3 },
          { token: 'is', weight: 0.05 },
        ]}
      />,
    )
    const highlighted = screen.getAllByTitle(/^is: 30\.0%/)
    expect(highlighted).toHaveLength(2)
  })

  it('leaves the text unhighlighted when every weight is zero', () => {
    const { container } = render(
      <HighlightedSentence text={REVIEW} attention={[{ token: 'dim', weight: 0 }]} />,
    )
    expect(container.querySelector('[title]')).toBeNull()
    expect(container.textContent).toBe(REVIEW)
  })

  it('survives an empty attention list', () => {
    const { container } = render(<HighlightedSentence text={REVIEW} attention={[]} />)
    expect(container.textContent).toBe(REVIEW)
  })

  it('does not highlight a word the model never reported', () => {
    render(<HighlightedSentence text={REVIEW} attention={[{ token: 'dim', weight: 0.5 }]} />)
    expect(screen.queryByTitle(/^amazing:/)).not.toBeInTheDocument()
  })
})

describe('PerClassTable', () => {
  const perClass = {
    negative: { f1: 0.734, precision: 0.645, recall: 0.8516, support: 128 },
    neutral: { f1: 0.6169, precision: 0.6835, recall: 0.5621, support: 169 },
    positive: { f1: 0.8912, precision: 0.9061, recall: 0.8768, support: 341 },
  }

  it('renders a row per label in fixed class-id order', () => {
    render(<PerClassTable perClass={perClass} />)
    const rows = screen.getAllByRole('row').slice(1) // drop the header
    expect(rows.map((row) => within(row).getByRole('rowheader').textContent)).toEqual([
      'negative',
      'neutral',
      'positive',
    ])
  })

  it('keeps that order even when the JSON arrives in another one', () => {
    // The guard against iterating Object.keys: a report serialised
    // differently must not reorder the table.
    const reversed = {
      positive: perClass.positive,
      neutral: perClass.neutral,
      negative: perClass.negative,
    }
    render(<PerClassTable perClass={reversed} />)
    const rows = screen.getAllByRole('row').slice(1)
    expect(rows.map((row) => within(row).getByRole('rowheader').textContent)).toEqual([
      'negative',
      'neutral',
      'positive',
    ])
  })

  it('shows a row of dashes for a label the report omits', () => {
    render(<PerClassTable perClass={{ negative: perClass.negative }} />)
    const rows = screen.getAllByRole('row').slice(1)
    const neutralRow = rows.find(
      (row) => within(row).getByRole('rowheader').textContent === 'neutral',
    )
    expect(neutralRow).toBeDefined()
    expect(within(neutralRow as HTMLElement).getAllByText('—').length).toBeGreaterThan(0)
  })

  it('prints the exact figures beside the bars', () => {
    render(<PerClassTable perClass={perClass} />)
    expect(screen.getAllByText('61.7%').length).toBeGreaterThan(0)
    expect(screen.getByText('341')).toBeInTheDocument()
  })
})

describe('ConfusionMatrix', () => {
  // absa-v1's recorded matrix, rows = gold.
  const matrix = [
    [109, 14, 5],
    [48, 95, 26],
    [12, 30, 299],
  ]

  it('prints every count, so colour is a second channel not the only one', () => {
    render(<ConfusionMatrix matrix={matrix} />)
    for (const count of matrix.flat()) {
      expect(screen.getAllByText(String(count)).length).toBeGreaterThan(0)
    }
  })

  it('labels rows as gold and columns as predicted', () => {
    render(<ConfusionMatrix matrix={matrix} />)
    expect(screen.getByText('Predicted')).toBeInTheDocument()
    const rowHeaders = screen.getAllByRole('rowheader').map((h) => h.textContent)
    expect(rowHeaders).toEqual(['negative', 'neutral', 'positive'])
  })

  it('totals each gold row', () => {
    render(<ConfusionMatrix matrix={matrix} />)
    expect(screen.getByText('128')).toBeInTheDocument() // 109 + 14 + 5
    expect(screen.getByText('169')).toBeInTheDocument()
    expect(screen.getByText('341')).toBeInTheDocument()
  })

  it('describes each cell as a share of its own row', () => {
    render(<ConfusionMatrix matrix={matrix} />)
    // 48 of 169 gold neutrals predicted negative: the project's central error.
    expect(
      screen.getByTitle('48 neutral examples predicted negative — 28.4% of all neutral'),
    ).toBeInTheDocument()
  })

  it('says "example" in the singular for a count of one', () => {
    render(<ConfusionMatrix matrix={[[1, 0, 0], [0, 0, 0], [0, 0, 0]]} />)
    expect(screen.getByTitle(/^1 negative example predicted negative/)).toBeInTheDocument()
  })

  it('renders a ragged or short matrix as zeros instead of throwing', () => {
    expect(() => render(<ConfusionMatrix matrix={[[1]]} />)).not.toThrow()
    expect(screen.getAllByText('0').length).toBeGreaterThan(0)
  })

  it('handles an all-zero row without dividing by zero', () => {
    render(<ConfusionMatrix matrix={[[0, 0, 0], [0, 0, 0], [0, 0, 0]]} />)
    expect(screen.getAllByText('0').length).toBeGreaterThan(8)
  })
})
