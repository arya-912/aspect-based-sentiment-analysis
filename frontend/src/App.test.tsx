/**
 * The app shell's sections.
 *
 * Which section is open lives in the URL fragment, so the back button works
 * and a section can be linked to. With four sections now, the thing worth
 * pinning is that an unknown fragment falls back rather than rendering
 * nothing.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import { getModelMeta } from './services/api'

vi.mock('./services/api', async () => {
  const actual = await vi.importActual<typeof import('./services/api')>('./services/api')
  return {
    ...actual,
    getModelMeta: vi.fn(),
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', model_loaded: true }),
    getProducts: vi.fn().mockResolvedValue([]),
    getCatalogueInsights: vi.fn().mockResolvedValue({
      aspects: [],
      products: {},
      model_version: 'absa-test',
    }),
  }
})

const mockedGetModelMeta = vi.mocked(getModelMeta)

function goTo(hash: string) {
  window.location.hash = hash
}

beforeEach(() => {
  mockedGetModelMeta.mockResolvedValue({
    version: 'absa-joint-v1',
    base_model: 'bert-base-uncased',
    task: 'aspect-based-sentiment-analysis',
    labels: ['negative', 'neutral', 'positive'],
    corpus: 'Joint_Train',
    evaluation: null,
  })
})

afterEach(() => {
  goTo('')
})

describe('sections', () => {
  it('offers all four', () => {
    goTo('')
    render(<App />)
    const nav = screen.getByRole('navigation', { name: /sections/i })
    expect(nav.textContent).toBe('AnalysisInsightsStorePerformance')
  })

  it('links each one by fragment, as real navigation', () => {
    render(<App />)
    const nav = screen.getByRole('navigation', { name: /sections/i })
    const hrefs = [...nav.querySelectorAll('a')].map((a) => a.getAttribute('href'))
    expect(hrefs).toEqual(['#analysis', '#insights', '#store', '#performance'])
  })

  it('opens Analysis by default', () => {
    goTo('')
    render(<App />)
    expect(screen.getByRole('link', { name: 'Analysis' })).toHaveAttribute(
      'aria-current',
      'page',
    )
  })

  it.each([
    ['#insights', 'Review insights'],
    ['#store', 'Demo store'],
  ])('renders the page for %s', (hash, heading) => {
    goTo(hash)
    render(<App />)
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument()
  })

  it('falls back to Analysis for an unknown fragment', () => {
    // A stale or hand-typed link must not render a blank main area.
    goTo('#nonsense')
    render(<App />)
    expect(screen.getByRole('link', { name: 'Analysis' })).toHaveAttribute(
      'aria-current',
      'page',
    )
  })

  it('follows a fragment change without a reload', async () => {
    goTo('#analysis')
    render(<App />)
    goTo('#store')
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Demo store' })).toBeInTheDocument(),
    )
  })
})

describe('copy that follows the served artifact', () => {
  it('names the pooled model’s subject once /model answers', async () => {
    goTo('')
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText(/laptop and restaurant reviews/i)).toBeInTheDocument(),
    )
  })

  it('says something true of every artifact while /model is unanswered', () => {
    mockedGetModelMeta.mockReturnValue(new Promise(() => {}))
    goTo('')
    render(<App />)
    expect(screen.getByText('One aspect at a time')).toBeInTheDocument()
  })

  it('does not break when /model fails', async () => {
    mockedGetModelMeta.mockRejectedValue(new Error('down'))
    goTo('')
    render(<App />)
    // Generic copy, and the sections still work: this only decides wording.
    expect(screen.getByText('One aspect at a time')).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.getByRole('navigation', { name: /sections/i })).toBeInTheDocument(),
    )
  })
})
