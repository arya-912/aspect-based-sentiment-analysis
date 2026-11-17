/**
 * The demo storefront.
 *
 * What these pin, in rough order of importance:
 *
 * - Ranking uses the **Wilson lower bound**, not the raw positive share, so
 *   3/3 does not outrank 9/10. That was the bug; this is the regression test.
 * - Every score is shown with its `n`.
 * - The weighted "best for me" ranking, and that it overrides the
 *   single-aspect sort rather than fighting it.
 * - The honesty note says which data is being shown -- demo or upload.
 * - The shelf renders before the slow background scoring finishes.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import StorePage from './StorePage'
import {
  ApiRequestError,
  getCatalogueInsights,
  getCatalogueStatus,
  getProduct,
  getProducts,
  resetCatalogue,
  uploadCatalogue,
} from '../services/api'
import type {
  CatalogueInsights,
  CatalogueStatus,
  ProductDetail,
  ProductSummary,
} from '../types'

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api')
  return {
    ...actual,
    getProducts: vi.fn(),
    getProduct: vi.fn(),
    getCatalogueInsights: vi.fn(),
    getCatalogueStatus: vi.fn(),
    uploadCatalogue: vi.fn(),
    resetCatalogue: vi.fn(),
  }
})

const mockedGetProducts = vi.mocked(getProducts)
const mockedGetProduct = vi.mocked(getProduct)
const mockedGetCatalogue = vi.mocked(getCatalogueInsights)
const mockedGetStatus = vi.mocked(getCatalogueStatus)
const mockedUpload = vi.mocked(uploadCatalogue)
const mockedReset = vi.mocked(resetCatalogue)

function product(over: Partial<ProductSummary> = {}): ProductSummary {
  return {
    slug: 'aurora-14',
    name: 'Aurora 14',
    category: 'laptop',
    subtitle: 'Ultraportable, 14-inch',
    price: '$1,099',
    review_count: 13,
    rating: null,
    rating_count: 0,
    reviews_seen: 0,
    ...over,
  }
}

const PRODUCTS: ProductSummary[] = [
  product(),
  product({
    slug: 'meridian-pro-16',
    name: 'Meridian Pro 16',
    subtitle: 'Workstation, 16-inch',
    price: '$2,399',
  }),
  product({
    slug: 'olive-and-ash',
    name: 'Olive & Ash',
    category: 'restaurant',
    subtitle: 'Mediterranean',
    price: '$$$',
    review_count: 15,
  }),
]

/**
 * The ranking fixture.
 *
 * `meridian-pro-16` has a *lower* positive share on battery life (0.90) than
 * `aurora-14` (1.00) but far more evidence (9/10 vs 1/1), so the Wilson
 * bound ranks it first. Raw-share ranking would put Aurora first, which is
 * exactly the bug.
 */
const CATALOGUE: CatalogueInsights = {
  aspects: ['battery life', 'screen', 'food'],
  model_version: 'absa-joint-v1',
  scored: 3,
  total: 3,
  scoring: false,
  products: {
    'aurora-14': [
      { aspect: 'battery life', positive_share: 1, negative_share: 0, total: 1, score: 0.207 },
      // Aurora leads on screen, so the two aspects give different orderings
      // -- which is the whole claim of the page.
      { aspect: 'screen', positive_share: 0.88, negative_share: 0.12, total: 8, score: 0.53 },
    ],
    'meridian-pro-16': [
      { aspect: 'battery life', positive_share: 0.9, negative_share: 0.1, total: 10, score: 0.596 },
      { aspect: 'screen', positive_share: 0.25, negative_share: 0.75, total: 4, score: 0.07 },
    ],
    'olive-and-ash': [
      { aspect: 'food', positive_share: 0.8, negative_share: 0.2, total: 5, score: 0.38 },
    ],
  },
}

const DEMO_STATUS: CatalogueStatus = {
  source: 'demo',
  source_name: 'built-in demo catalogue',
  products: 3,
  scored: 3,
  scoring: false,
  model_version: 'absa-joint-v1',
  rows_used: 0,
  products_found: 0,
  truncated_products: false,
  truncated_reviews: false,
  note: '',
}

const UPLOAD_STATUS: CatalogueStatus = {
  ...DEMO_STATUS,
  source: 'upload',
  source_name: 'yelp.csv',
  products: 24,
  scored: 4,
  rows_used: 14351,
  products_found: 2526,
  truncated_products: true,
  truncated_reviews: true,
  note: 'These numbers come from an uploaded file, not the built-in demo catalogue.',
}

const DETAIL: ProductDetail = {
  ...PRODUCTS[0],
  reviews: ['Battery life is amazing but the screen is dim.'],
  insights: {
    reviews: 1,
    sentences: 1,
    predictions: 2,
    mixed_sentences: 1,
    mixed_aspects: 0,
    truncated: false,
    model_version: 'absa-joint-v1',
    aspects: [
      {
        aspect: 'battery life',
        source: 'given',
        total: 4,
        counts: { positive: 4 },
        shares: { positive: 1 },
        dominant: 'positive',
        mean_confidence: 0.99,
        mixed: false,
        examples: [],
      },
      {
        aspect: 'screen',
        source: 'given',
        total: 5,
        counts: { negative: 4, positive: 1 },
        shares: { negative: 0.8, positive: 0.2 },
        dominant: 'negative',
        mean_confidence: 0.95,
        mixed: true,
        examples: [],
      },
    ],
  },
}

beforeEach(() => {
  vi.clearAllMocks()
  mockedGetProducts.mockResolvedValue(PRODUCTS)
  mockedGetCatalogue.mockResolvedValue(CATALOGUE)
  mockedGetStatus.mockResolvedValue(DEMO_STATUS)
  mockedGetProduct.mockResolvedValue(DETAIL)
})

/** Product names in the order they appear on the shelf. */
function shelfOrder(): string[] {
  return screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent ?? '')
}

async function renderShelf() {
  const user = userEvent.setup()
  render(<StorePage />)
  await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
  return { user }
}

async function sortBy(user: ReturnType<typeof userEvent.setup>, aspect: string) {
  const control = screen.getByRole('combobox', { name: /best for$/i })
  await waitFor(() => expect(control).toBeEnabled())
  await user.selectOptions(control, aspect)
}

describe('the shelf', () => {
  it('lists every product with its subtitle and price', async () => {
    await renderShelf()
    expect(shelfOrder()).toEqual(['Aurora 14', 'Meridian Pro 16', 'Olive & Ash'])
    expect(screen.getByText('Ultraportable, 14-inch')).toBeInTheDocument()
    expect(screen.getByText('$1,099')).toBeInTheDocument()
  })

  it('renders before the slow aspect scores arrive', async () => {
    mockedGetCatalogue.mockReturnValue(new Promise(() => {}))
    render(<StorePage />)
    await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
  })

  it('reports scoring progress while the background pass runs', async () => {
    mockedGetStatus.mockResolvedValue({ ...DEMO_STATUS, scoring: true, scored: 1, products: 3 })
    mockedGetCatalogue.mockResolvedValue({ ...CATALOGUE, scoring: true, scored: 1, total: 3 })
    render(<StorePage />)
    expect(await screen.findByText(/1 of 3 products done/i)).toBeInTheDocument()
  })

  it('starts in catalogue order, with no aspect chosen', async () => {
    await renderShelf()
    expect(screen.getByRole('combobox', { name: /best for$/i })).toHaveValue('')
  })
})

describe('ranking by one aspect', () => {
  it('uses the lower bound, so better evidence wins over a perfect score', async () => {
    // The regression test for the original bug. Aurora is 1/1 positive on
    // battery life (share 1.00); Meridian is 9/10 (share 0.90). Ranking on
    // share puts Aurora first; ranking on the bound puts Meridian first.
    const { user } = await renderShelf()
    await sortBy(user, 'battery life')
    expect(shelfOrder()[0]).toBe('Meridian Pro 16')
  })

  it('gives a different order for a different aspect', async () => {
    // One overall rating cannot express both orderings; that is the page.
    const { user } = await renderShelf()
    await sortBy(user, 'battery life')
    expect(shelfOrder()[0]).toBe('Meridian Pro 16')
    await sortBy(user, 'screen')
    expect(shelfOrder()[0]).toBe('Aurora 14')
  })

  it('shows the share, the n and the score it ranked on', async () => {
    const { user } = await renderShelf()
    await sortBy(user, 'battery life')
    expect(screen.getByText(/90% positive/)).toBeInTheDocument()
    expect(screen.getByText(/of 10 sentences/)).toBeInTheDocument()
    expect(screen.getByText(/Ranked on 0\.60/)).toBeInTheDocument()
  })

  it('says "1 sentence" rather than "1 sentences"', async () => {
    const { user } = await renderShelf()
    await sortBy(user, 'battery life')
    expect(screen.getByText(/of 1 sentence$/)).toBeInTheDocument()
  })

  it('ranks a product the aspect is not discussed on last, not out of sight', async () => {
    const { user } = await renderShelf()
    await sortBy(user, 'battery life')
    expect(shelfOrder()).toHaveLength(3)
    expect(shelfOrder()[2]).toBe('Olive & Ash')
    expect(screen.getByText(/not discussed on/i)).toBeInTheDocument()
  })

  it('says what the shelf is ranked by', async () => {
    const { user } = await renderShelf()
    await sortBy(user, 'screen')
    expect(screen.getByText(/on the\s+lower bound of its positive share/i)).toBeInTheDocument()
  })
})

describe('"best for me" weights', () => {
  async function setWeight(
    user: ReturnType<typeof userEvent.setup>,
    aspect: string,
    step: number,
  ) {
    const group = screen.getByRole('radiogroup', { name: `Weight for ${aspect}` })
    await user.click(within(group).getByRole('radio', { name: `${aspect} weight ${step}` }))
  }

  it('reorders the shelf by the weighted mix', async () => {
    const { user } = await renderShelf()
    await setWeight(user, 'battery life', 3)
    expect(shelfOrder()[0]).toBe('Meridian Pro 16')
  })

  it('changes the order when the weights change', async () => {
    const { user } = await renderShelf()
    await setWeight(user, 'food', 3)
    // Only the restaurant is discussed on food, so it leads.
    expect(shelfOrder()[0]).toBe('Olive & Ash')
  })

  it('shows the match and how many weighted aspects it covers', async () => {
    const { user } = await renderShelf()
    await setWeight(user, 'battery life', 2)
    expect(screen.getAllByText(/Match for your weights/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/over 1 of 1 weighted aspect/).length).toBeGreaterThan(0)
  })

  it('says a product cannot be compared rather than scoring it zero', async () => {
    // Scoring an unreviewed product 0 would rank it below a disliked one.
    const { user } = await renderShelf()
    await setWeight(user, 'food', 1)
    expect(
      screen.getAllByText(/None of your weighted aspects is discussed/i).length,
    ).toBeGreaterThan(0)
  })

  it('takes over from the single-aspect sort', async () => {
    const { user } = await renderShelf()
    await sortBy(user, 'screen')
    await setWeight(user, 'battery life', 3)
    expect(screen.getByText(/Ranked by your weights/)).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /best for$/i })).toBeDisabled()
  })

  it('can be cleared', async () => {
    const { user } = await renderShelf()
    await setWeight(user, 'battery life', 3)
    await user.click(screen.getByRole('button', { name: /clear weights/i }))
    expect(screen.queryByText(/Ranked by your weights/)).not.toBeInTheDocument()
  })

  it('reports each weight in the ranking line', async () => {
    const { user } = await renderShelf()
    await setWeight(user, 'battery life', 3)
    expect(screen.getByText(/battery life ×3/)).toBeInTheDocument()
  })
})

describe('the star rating', () => {
  it('is shown when the catalogue carries one', async () => {
    mockedGetProducts.mockResolvedValue([
      product({ rating: 4.5, rating_count: 187 }),
      ...PRODUCTS.slice(1),
    ])
    await renderShelf()
    expect(screen.getByText('4.5')).toBeInTheDocument()
    expect(screen.getByText('(187 ratings)')).toBeInTheDocument()
  })

  it('is absent for the demo catalogue, whose sentences have no stars', async () => {
    await renderShelf()
    expect(screen.queryByText(/ratings\)/)).not.toBeInTheDocument()
  })
})

describe("a product's breakdown", () => {
  it('leads with a one-line pros and cons summary', async () => {
    const { user } = await renderShelf()
    await user.click(screen.getAllByRole('button', { name: /reviews, by aspect/i })[0])
    expect(await screen.findByText(/Praised:/)).toBeInTheDocument()
    expect(screen.getByText(/Complained about:/)).toBeInTheDocument()
  })

  it('offers the breakdown as a table too', async () => {
    const { user } = await renderShelf()
    await user.click(screen.getAllByRole('button', { name: /reviews, by aspect/i })[0])
    expect(await screen.findByText('Show as a table')).toBeInTheDocument()
  })

  it('says how many reviews the file held when it sampled them', async () => {
    mockedGetProducts.mockResolvedValue([
      product({ review_count: 15, reviews_seen: 187 }),
      ...PRODUCTS.slice(1),
    ])
    await renderShelf()
    expect(screen.getByText(/\(of 187\)/)).toBeInTheDocument()
  })

  it('collapses again when the same product is clicked twice', async () => {
    const { user } = await renderShelf()
    const toggle = screen.getAllByRole('button', { name: /reviews, by aspect/i })[0]
    await user.click(toggle)
    await waitFor(() => expect(toggle).toHaveAttribute('aria-expanded', 'true'))
    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
  })
})

describe('uploading a catalogue', () => {
  const file = () =>
    new File(['product_name,category,review_text\nA,laptop,The screen is dim.\n'], 'yelp.csv', {
      type: 'text/csv',
    })

  it('offers an upload button and a sample CSV', async () => {
    await renderShelf()
    expect(screen.getByRole('button', { name: /upload catalogue/i })).toBeInTheDocument()
    const sample = screen.getByRole('link', { name: /sample csv/i })
    expect(sample).toHaveAttribute('download', 'absa-catalogue-sample.csv')
  })

  it('sends the chosen category with the file', async () => {
    mockedUpload.mockResolvedValue({ ...UPLOAD_STATUS, warnings: [] })
    const { user } = await renderShelf()
    await user.selectOptions(
      screen.getByRole('combobox', { name: /these reviews are about/i }),
      'laptop',
    )
    await user.upload(screen.getByLabelText(/upload a catalogue csv/i), file())
    await waitFor(() => expect(mockedUpload).toHaveBeenCalled())
    expect(mockedUpload.mock.calls[0][1]).toBe('laptop')
  })

  it('summarises what was loaded', async () => {
    mockedUpload.mockResolvedValue({
      ...UPLOAD_STATUS,
      warnings: ['The file held 2,526 products; the 24 most-reviewed are shown.'],
    })
    const { user } = await renderShelf()
    await user.upload(screen.getByLabelText(/upload a catalogue csv/i), file())

    expect(await screen.findByText(/24 products of 2,526/)).toBeInTheDocument()
    expect(screen.getByText(/most-reviewed are shown/)).toBeInTheDocument()
  })

  it("surfaces the backend's own rejection message verbatim", async () => {
    // The 422 names the column or row to fix; rewording it would throw away
    // the only actionable part.
    mockedUpload.mockRejectedValue(
      new ApiRequestError('missing required column(s): product_name, review_text', 422),
    )
    const { user } = await renderShelf()
    await user.upload(screen.getByLabelText(/upload a catalogue csv/i), file())

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'missing required column(s): product_name, review_text',
    )
  })

  it('refetches the shelf after a successful upload', async () => {
    mockedUpload.mockResolvedValue({ ...UPLOAD_STATUS, warnings: [] })
    const { user } = await renderShelf()
    mockedGetProducts.mockClear()
    await user.upload(screen.getByLabelText(/upload a catalogue csv/i), file())
    await waitFor(() => expect(mockedGetProducts).toHaveBeenCalled())
  })

  it('offers a way back to the demo catalogue only once something is uploaded', async () => {
    await renderShelf()
    expect(
      screen.queryByRole('button', { name: /back to the demo catalogue/i }),
    ).not.toBeInTheDocument()

    mockedGetStatus.mockResolvedValue(UPLOAD_STATUS)
    mockedReset.mockResolvedValue(DEMO_STATUS)
    const { user } = await renderShelf()
    const back = await screen.findByRole('button', { name: /back to the demo catalogue/i })
    await user.click(back)
    await waitFor(() => expect(mockedReset).toHaveBeenCalled())
  })
})

describe('the honesty note', () => {
  it('describes the demo catalogue when that is what is served', async () => {
    await renderShelf()
    expect(screen.getByText(/the products are invented; the reviews/i)).toBeInTheDocument()
    expect(screen.getByText(/which flatters it/i)).toBeInTheDocument()
  })

  it('names the uploaded file instead, and does not claim the demo caveats', async () => {
    mockedGetStatus.mockResolvedValue(UPLOAD_STATUS)
    await renderShelf()
    expect(screen.getByText(/uploaded by you/i)).toBeInTheDocument()
    expect(screen.getByText('yelp.csv')).toBeInTheDocument()
    expect(screen.queryByText(/the products are invented/i)).not.toBeInTheDocument()
  })

  it('says an upload is not persisted', async () => {
    mockedGetStatus.mockResolvedValue(UPLOAD_STATUS)
    await renderShelf()
    expect(screen.getByText(/restart returns to the\s+demo catalogue/i)).toBeInTheDocument()
  })

  it('states the vocabulary limit either way', async () => {
    await renderShelf()
    expect(screen.getByText(/covers laptops and restaurants only/i)).toBeInTheDocument()
  })

  it('explains that ranking uses a lower bound', async () => {
    await renderShelf()
    expect(screen.getByText(/cannot outrank one with\s+thirty on luck alone/i)).toBeInTheDocument()
  })
})

describe('failure', () => {
  it('reports a catalogue that cannot be loaded', async () => {
    mockedGetProducts.mockRejectedValue(new ApiRequestError('model not loaded', 503))
    render(<StorePage />)
    expect(await screen.findByRole('alert')).toHaveTextContent('model not loaded')
  })

  it('still shows the shelf when only the scoring failed', async () => {
    mockedGetCatalogue.mockRejectedValue(new ApiRequestError('model not loaded', 503))
    render(<StorePage />)
    await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
