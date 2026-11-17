/**
 * `StarRating` and `ProsCons`.
 *
 * Both exist to be read beside the aspect bars, not instead of them, and
 * both have to be legible without colour: the rating prints its number and
 * the summary is words.
 */

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ProsCons from './ProsCons'
import StarRating from './StarRating'
import type { AspectSummary } from '../types'

function aspect(over: Partial<AspectSummary> = {}): AspectSummary {
  return {
    aspect: 'food',
    source: 'given',
    total: 5,
    counts: { positive: 5 },
    shares: { positive: 1 },
    dominant: 'positive',
    mean_confidence: 0.9,
    mixed: false,
    examples: [],
    ...over,
  }
}

describe('StarRating', () => {
  it('prints the number, not only the glyphs', () => {
    render(<StarRating rating={4.15} count={187} />)
    expect(screen.getByText('4.2')).toBeInTheDocument()
  })

  it('says how many ratings it averages', () => {
    render(<StarRating rating={4.5} count={187} />)
    expect(screen.getByText('(187 ratings)')).toBeInTheDocument()
  })

  it('uses the singular for one rating', () => {
    render(<StarRating rating={3} count={1} />)
    expect(screen.getByText('(1 rating)')).toBeInTheDocument()
  })

  it('hides the glyphs from assistive tech, since the number carries it', () => {
    const { container } = render(<StarRating rating={4} count={10} />)
    expect(container.querySelector('[aria-hidden="true"]')).toBeInTheDocument()
  })

  it.each([0, 2.5, 5])('renders %s without throwing', (rating) => {
    expect(() => render(<StarRating rating={rating} count={3} />)).not.toThrow()
  })
})

describe('ProsCons', () => {
  it('names a clearly praised aspect', () => {
    render(<ProsCons aspects={[aspect({ aspect: 'keyboard' })]} />)
    expect(screen.getByText(/Praised:/)).toBeInTheDocument()
    expect(screen.getByText(/keyboard/)).toBeInTheDocument()
  })

  it('names a clearly criticised aspect', () => {
    render(
      <ProsCons
        aspects={[
          aspect({
            aspect: 'price',
            counts: { negative: 4 },
            shares: { negative: 1 },
            dominant: 'negative',
          }),
        ]}
      />,
    )
    expect(screen.getByText(/Complained about:/)).toBeInTheDocument()
  })

  it('reports both in one line', () => {
    render(
      <ProsCons
        aspects={[
          aspect({ aspect: 'keyboard' }),
          aspect({
            aspect: 'price',
            counts: { negative: 4 },
            shares: { negative: 1 },
            dominant: 'negative',
          }),
        ]}
      />,
    )
    expect(screen.getByText(/Praised:/)).toBeInTheDocument()
    expect(screen.getByText(/Complained about:/)).toBeInTheDocument()
  })

  it('will not call one sentence a selling point', () => {
    // Below the evidence floor, so it is named in neither list.
    render(<ProsCons aspects={[aspect({ total: 1 })]} />)
    expect(screen.getByText(/No aspect here is clearly praised or criticised/i))
      .toBeInTheDocument()
  })

  it('leaves a genuinely split aspect out of both lists', () => {
    // Calling it a pro would be as wrong as calling it a con; the bars still
    // show it and the `mixed` tag marks it.
    render(
      <ProsCons
        aspects={[
          aspect({
            aspect: 'service',
            total: 10,
            counts: { positive: 5, negative: 5 },
            shares: { positive: 0.5, negative: 0.5 },
            mixed: true,
          }),
        ]}
      />,
    )
    expect(screen.getByText(/No aspect here is clearly praised or criticised/i))
      .toBeInTheDocument()
  })

  it('orders the praised aspects by how strongly they are praised', () => {
    render(
      <ProsCons
        aspects={[
          aspect({ aspect: 'food', shares: { positive: 0.7 }, counts: { positive: 7 }, total: 10 }),
          aspect({ aspect: 'decor', shares: { positive: 1 }, counts: { positive: 5 }, total: 5 }),
        ]}
      />,
    )
    expect(screen.getByText(/decor, food/)).toBeInTheDocument()
  })

  it('says something rather than nothing for an empty breakdown', () => {
    render(<ProsCons aspects={[]} />)
    expect(screen.getByText(/No aspect here is clearly praised or criticised/i))
      .toBeInTheDocument()
  })
})
