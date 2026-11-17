/**
 * The aspect-insight components: `AspectShareBar`, `SentimentLegend`,
 * `AspectInsightRow`, `AspectTable`.
 *
 * The share bar is a chart, and the rules it has to keep are the ones that
 * make a chart readable without color: the figures exist as text, the parts
 * are named in a legend and in the bar's accessible name, a zero draws
 * nothing, and the same numbers are available as a table.
 */

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import AspectInsightRow from './AspectInsightRow'
import AspectShareBar from './AspectShareBar'
import AspectTable from './AspectTable'
import SentimentLegend from './SentimentLegend'
import type { AspectSummary } from '../types'

const SUMMARY: AspectSummary = {
  aspect: 'screen',
  source: 'corpus',
  total: 7,
  counts: { negative: 5, positive: 2 },
  shares: { negative: 0.7143, positive: 0.2857 },
  dominant: 'negative',
  mean_confidence: 0.9412,
  mixed: true,
  examples: [
    {
      sentence: 'Battery life is amazing but the screen is dim.',
      surface: 'screen',
      sentiment: 'negative',
      confidence: 0.9977,
      review_index: 0,
    },
    {
      sentence: 'The screen is gorgeous.',
      surface: 'screen',
      sentiment: 'positive',
      confidence: 0.9997,
      review_index: 1,
    },
  ],
}

describe('AspectShareBar', () => {
  it('names every part and its figures in its accessible name', () => {
    // The bar is a graphic; without this it tells a screen reader nothing.
    render(
      <AspectShareBar
        aspect="screen"
        shares={{ negative: 0.7143, positive: 0.2857 }}
        counts={{ negative: 5, positive: 2 }}
      />,
    )
    const bar = screen.getByRole('img')
    expect(bar).toHaveAccessibleName(
      'screen: 5 negative (71%), 0 neutral (0%), 2 positive (29%)',
    )
  })

  it('draws one segment per present label, sized by share', () => {
    const { container } = render(
      <AspectShareBar
        aspect="screen"
        shares={{ negative: 0.75, positive: 0.25 }}
        counts={{ negative: 3, positive: 1 }}
      />,
    )
    const segments = container.querySelectorAll('[title]')
    expect(segments).toHaveLength(2)
    expect((segments[0] as HTMLElement).style.width).toBe('75%')
    expect((segments[1] as HTMLElement).style.width).toBe('25%')
  })

  it('draws nothing for a label at zero', () => {
    // A sliver for an absent label reads as a real quantity.
    const { container } = render(
      <AspectShareBar
        aspect="screen"
        shares={{ negative: 1, neutral: 0, positive: 0 }}
        counts={{ negative: 4, neutral: 0, positive: 0 }}
      />,
    )
    expect(container.querySelectorAll('[title]')).toHaveLength(1)
  })

  it('orders segments negative, neutral, positive', () => {
    // Class-id order, so the bar does not re-sort itself between aspects.
    const { container } = render(
      <AspectShareBar
        aspect="food"
        shares={{ positive: 0.5, neutral: 0.25, negative: 0.25 }}
        counts={{ positive: 2, neutral: 1, negative: 1 }}
      />,
    )
    const titles = [...container.querySelectorAll('[title]')].map((node) =>
      node.getAttribute('title'),
    )
    expect(titles).toEqual(['1 negative (25%)', '1 neutral (25%)', '2 positive (50%)'])
  })

  it('gives every segment hover text with its count and share', () => {
    render(
      <AspectShareBar
        aspect="screen"
        shares={{ negative: 0.7143, positive: 0.2857 }}
        counts={{ negative: 5, positive: 2 }}
      />,
    )
    expect(screen.getByTitle('5 negative (71%)')).toBeInTheDocument()
    expect(screen.getByTitle('2 positive (29%)')).toBeInTheDocument()
  })

  it('survives an empty distribution', () => {
    const { container } = render(
      <AspectShareBar aspect="screen" shares={{}} counts={{}} />,
    )
    expect(container.querySelectorAll('[title]')).toHaveLength(0)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })
})

describe('SentimentLegend', () => {
  it('names all three labels', () => {
    render(<SentimentLegend />)
    for (const label of ['negative', 'neutral', 'positive']) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  })

  it('hides its swatches from assistive tech, since the text carries the name', () => {
    const { container } = render(<SentimentLegend />)
    expect(container.querySelectorAll('[aria-hidden="true"]')).toHaveLength(3)
  })
})

describe('AspectInsightRow', () => {
  /**
   * A sentiment badge, found by its dot.
   *
   * The label text is ambiguous here by design: "negative" is the dominant
   * badge, a figure in the row, *and* the badge on each example. Matching on
   * the dot picks out the badges specifically.
   */
  function badges(label: string): HTMLElement[] {
    return screen
      .getAllByText(label)
      .filter((node) => node.querySelector('[aria-hidden="true"]') !== null)
  }

  it('shows the aspect, its sentence count and its dominant label', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(screen.getByText('screen')).toBeInTheDocument()
    expect(screen.getByText('7 sentences')).toBeInTheDocument()
    // The dominant badge, plus one on the negative example.
    expect(badges('negative').length).toBeGreaterThan(0)
  })

  it('prints the figures as text, not only as bar widths', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(screen.getByText(/71%/)).toBeInTheDocument()
    expect(screen.getByText(/29%/)).toBeInTheDocument()
  })

  it('marks a mixed aspect as mixed', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(screen.getByText('mixed')).toBeInTheDocument()
  })

  it('does not call a single-polarity aspect mixed', () => {
    render(
      <AspectInsightRow
        summary={{
          ...SUMMARY,
          mixed: false,
          counts: { negative: 7 },
          shares: { negative: 1 },
          examples: [],
        }}
      />,
    )
    expect(screen.queryByText('mixed')).not.toBeInTheDocument()
  })

  it('opens the evidence by default for a mixed aspect', () => {
    // The disagreement is the finding; collapsing it buries the lede.
    const { container } = render(<AspectInsightRow summary={SUMMARY} />)
    expect(container.querySelector('details')).toHaveAttribute('open')
  })

  it('leaves the evidence collapsed when the aspect agrees with itself', () => {
    const { container } = render(
      <AspectInsightRow summary={{ ...SUMMARY, mixed: false }} />,
    )
    expect(container.querySelector('details')).not.toHaveAttribute('open')
  })

  it('quotes each example with the wording the model was asked about', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(
      screen.getByText('Battery life is amazing but the screen is dim.'),
    ).toBeInTheDocument()
    expect(screen.getAllByText(/about “screen”/).length).toBeGreaterThan(0)
  })

  it('numbers the review an example came from from one, not zero', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(screen.getByText(/review 1/)).toBeInTheDocument()
    expect(screen.getByText(/review 2/)).toBeInTheDocument()
  })

  it('says where the aspect came from, since it did not come from the model', () => {
    render(<AspectInsightRow summary={SUMMARY} />)
    expect(screen.getByText(/from the corpus vocabulary/)).toBeInTheDocument()
  })

  it.each([
    ['corpus-rare', /rare corpus term/],
    ['given', /you named it/],
  ] as const)('describes the %s source in words', (source, pattern) => {
    render(<AspectInsightRow summary={{ ...SUMMARY, source }} />)
    expect(screen.getByText(pattern)).toBeInTheDocument()
  })

  it('omits the source on the storefront, where every aspect is a sort key', () => {
    render(<AspectInsightRow summary={SUMMARY} showSource={false} />)
    expect(screen.queryByText(/from the corpus vocabulary/)).not.toBeInTheDocument()
  })

  it('renders an aspect with no examples', () => {
    const { container } = render(
      <AspectInsightRow summary={{ ...SUMMARY, examples: [] }} />,
    )
    expect(container.querySelector('details')).toBeNull()
    expect(screen.getByText('screen')).toBeInTheDocument()
  })
})

describe('AspectTable', () => {
  it('gives every bar’s numbers in a readable table', () => {
    render(<AspectTable aspects={[SUMMARY]} />)
    const row = screen.getAllByRole('row')[1]
    expect(within(row).getByRole('rowheader')).toHaveTextContent('screen')
    expect(row).toHaveTextContent('71%')
    expect(row).toHaveTextContent('29%')
    expect(row).toHaveTextContent('7')
  })

  it('shows a dash for a label with no sentences', () => {
    render(<AspectTable aspects={[SUMMARY]} />)
    // Neutral is absent from this aspect.
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('keeps a fixed column order rather than following the JSON', () => {
    render(<AspectTable aspects={[SUMMARY]} />)
    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent)
    expect(headers).toEqual(['Aspect', 'negative', 'neutral', 'positive', 'Sentences'])
  })

  it('carries a caption for a screen reader', () => {
    render(<AspectTable aspects={[SUMMARY]} />)
    expect(screen.getByRole('table')).toHaveAccessibleName(/per aspect/i)
  })

  it('renders an empty aspect list without a body row', () => {
    render(<AspectTable aspects={[]} />)
    expect(screen.getAllByRole('row')).toHaveLength(1)
  })
})
