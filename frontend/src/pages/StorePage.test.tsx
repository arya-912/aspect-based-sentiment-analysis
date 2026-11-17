/**
 * The demo storefront.
 *
 * The behaviour that carries the demo: choosing an aspect reorders the shelf,
 * and a different aspect gives a different order. That is the claim -- one
 * overall rating cannot express both -- so it is asserted directly rather
 * than left to a screenshot.
 *
 * Also pinned: the shelf renders before the (slow) scores arrive, a product
 * the aspect is not discussed on is ranked last rather than hidden, and the
 * page says which reviews the model was trained on.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import StorePage from './StorePage'
import {
  ApiRequestError,
  getCatalogueInsights,
  getProduct,
  getProducts,
} from '../services/api'
import type { CatalogueInsights, ProductDetail, ProductSummary } from '../types'

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api')
  return {
    ...actual,
    getProducts: vi.fn(),
    getProduct: vi.fn(),
    getCatalogueInsights: vi.fn(),
  }
})

const mockedGetProducts = vi.mocked(getProducts)
const mockedGetProduct = vi.mocked(getProduct)
const mockedGetCatalogue = vi.mocked(getCatalogueInsights)

const PRODUCTS: ProductSummary[] = [
  {
    slug: 'aurora-14',
    name: 'Aurora 14',
    category: 'laptop',
    subtitle: 'Ultraportable, 14-inch',
    price: '$1,099',
    review_count: 13,
  },
  {
    slug: 'meridian-pro-16',
    name: 'Meridian Pro 16',
    category: 'laptop',
    subtitle: 'Workstation, 16-inch',
    price: '$2,399',
    review_count: 13,
  },
  {
    slug: 'olive-and-ash',
    name: 'Olive & Ash',
    category: 'restaurant',
    subtitle: 'Mediterranean, mid-town',
    price: '$$$',
    review_count: 15,
  },
]

/**
 * Aurora wins on battery, Meridian wins on screen. The orderings differ,
 * which is the point of the page.
 */
const CATALOGUE: CatalogueInsights = {
  aspects: ['battery life', 'screen', 'food'],
  model_version: 'absa-joint-v1',
  products: {
    'aurora-14': [
      { aspect: 'battery life', positive_share: 1, negative_share: 0, total: 4 },
      { aspect: 'screen', positive_share: 0.29, negative_share: 0.71, total: 7 },
    ],
    'meridian-pro-16': [
      { aspect: 'battery life', positive_share: 0, negative_share: 1, total: 3 },
      { aspect: 'screen', positive_share: 1, negative_share: 0, total: 4 },
    ],
    'olive-and-ash': [
      { aspect: 'food', positive_share: 0.8, negative_share: 0.2, total: 5 },
    ],
  },
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
        total: 1,
        counts: { positive: 1 },
        shares: { positive: 1 },
        dominant: 'positive',
        mean_confidence: 0.99,
        mixed: false,
        examples: [],
      },
    ],
  },
}

beforeEach(() => {
  mockedGetProducts.mockReset()
  mockedGetProduct.mockReset()
  mockedGetCatalogue.mockReset()
  mockedGetProducts.mockResolvedValue(PRODUCTS)
  mockedGetCatalogue.mockResolvedValue(CATALOGUE)
  mockedGetProduct.mockResolvedValue(DETAIL)
})

/** Product names in the order they appear on the shelf. */
function shelfOrder(): string[] {
  return screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent ?? '')
}

/**
 * The breakdown toggle on the first card.
 *
 * By position, not by name: two fixture products have 13 reviews, so the
 * accessible name is not unique and never will be on a real shelf either.
 */
function firstBreakdownToggle(): HTMLElement {
  return screen.getAllByRole('button', { name: /reviews, by aspect/i })[0]
}

async function renderShelf() {
  const user = userEvent.setup()
  render(<StorePage />)
  await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
  return { user }
}

describe('the shelf', () => {
  it('lists every product with its subtitle and price', async () => {
    await renderShelf()
    expect(shelfOrder()).toEqual(['Aurora 14', 'Meridian Pro 16', 'Olive & Ash'])
    expect(screen.getByText('Ultraportable, 14-inch')).toBeInTheDocument()
    expect(screen.getByText('$1,099')).toBeInTheDocument()
  })

  it('renders before the slow aspect scores arrive', async () => {
    // Two requests on purpose: the shelf must not wait ~8s for the scoring.
    mockedGetCatalogue.mockReturnValue(new Promise(() => {}))
    render(<StorePage />)
    await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
    expect(screen.getByRole('status')).toHaveTextContent(/scoring every review/i)
  })

  it('disables the sort control until the scores land', async () => {
    mockedGetCatalogue.mockReturnValue(new Promise(() => {}))
    render(<StorePage />)
    await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
    expect(screen.getByRole('combobox', { name: /best for/i })).toBeDisabled()
  })

  it('enables it once they have', async () => {
    await renderShelf()
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /best for/i })).toBeEnabled(),
    )
  })

  it('starts in catalogue order, with no aspect chosen', async () => {
    await renderShelf()
    expect(screen.getByRole('combobox', { name: /best for/i })).toHaveValue('')
  })
})

describe('sorting by one aspect', () => {
  it('reorders the shelf', async () => {
    const { user } = await renderShelf()
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /best for/i })).toBeEnabled(),
    )
    await user.selectOptions(
      screen.getByRole('combobox', { name: /best for/i }),
      'battery life',
    )
    expect(shelfOrder()[0]).toBe('Aurora 14')
  })

  it('gives a different order for a different aspect', async () => {
    // The whole argument of the page: one overall rating cannot express both.
    const { user } = await renderShelf()
    const control = screen.getByRole('combobox', { name: /best for/i })
    await waitFor(() => expect(control).toBeEnabled())

    await user.selectOptions(control, 'battery life')
    const byBattery = shelfOrder()
    await user.selectOptions(control, 'screen')
    const byScreen = shelfOrder()

    expect(byBattery[0]).toBe('Aurora 14')
    expect(byScreen[0]).toBe('Meridian Pro 16')
    expect(byBattery).not.toEqual(byScreen)
  })

  it('shows the chosen aspect’s share and count on each card', async () => {
    const { user } = await renderShelf()
    const control = screen.getByRole('combobox', { name: /best for/i })
    await waitFor(() => expect(control).toBeEnabled())
    await user.selectOptions(control, 'battery life')

    expect(screen.getByText(/100% positive/)).toBeInTheDocument()
    expect(screen.getByText(/of 4/)).toBeInTheDocument()
  })

  it('ranks a product the aspect is not discussed on last, not out of sight', async () => {
    const { user } = await renderShelf()
    const control = screen.getByRole('combobox', { name: /best for/i })
    await waitFor(() => expect(control).toBeEnabled())
    await user.selectOptions(control, 'battery life')

    // The restaurant has no battery score, and is still on the shelf.
    expect(shelfOrder()).toHaveLength(3)
    expect(shelfOrder()[2]).toBe('Olive & Ash')
    expect(screen.getByText(/not discussed on/i)).toBeInTheDocument()
  })

  it('says what the shelf is ranked by', async () => {
    const { user } = await renderShelf()
    const control = screen.getByRole('combobox', { name: /best for/i })
    await waitFor(() => expect(control).toBeEnabled())
    await user.selectOptions(control, 'screen')
    expect(screen.getByText(/ranked by positive share for/i)).toBeInTheDocument()
  })
})

describe('filtering by category', () => {
  it('narrows the shelf', async () => {
    const { user } = await renderShelf()
    await user.selectOptions(
      screen.getByRole('combobox', { name: /category/i }),
      'restaurant',
    )
    expect(shelfOrder()).toEqual(['Olive & Ash'])
  })

  it('offers only aspects the surviving products are scored on', async () => {
    // "battery life" is a dead sort key once only restaurants are shown.
    const { user } = await renderShelf()
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /best for/i })).toBeEnabled(),
    )
    await user.selectOptions(
      screen.getByRole('combobox', { name: /category/i }),
      'restaurant',
    )
    const options = within(screen.getByRole('combobox', { name: /best for/i }))
      .getAllByRole('option')
      .map((option) => option.textContent)
    expect(options).toContain('food')
    expect(options).not.toContain('battery life')
  })

  it('resets the sort when the category changes', async () => {
    const { user } = await renderShelf()
    const control = screen.getByRole('combobox', { name: /best for/i })
    await waitFor(() => expect(control).toBeEnabled())
    await user.selectOptions(control, 'battery life')
    await user.selectOptions(screen.getByRole('combobox', { name: /category/i }), 'restaurant')
    expect(screen.getByRole('combobox', { name: /best for/i })).toHaveValue('')
  })
})

describe('a product’s breakdown', () => {
  it('is fetched on demand and shown per aspect', async () => {
    const { user } = await renderShelf()
    await user.click(firstBreakdownToggle())

    await waitFor(() => expect(mockedGetProduct).toHaveBeenCalledWith('aurora-14'))
    // Matched on the text node: the line is built from interpolations, so
    // "2 aspect judgements" is not one node.
    expect(await screen.findByText(/aspect judgements/)).toBeInTheDocument()
    expect(screen.getAllByText('battery life').length).toBeGreaterThan(0)
  })

  it('says it is working while the breakdown loads', async () => {
    mockedGetProduct.mockReturnValue(new Promise(() => {}))
    const { user } = await renderShelf()
    await user.click(firstBreakdownToggle())
    expect(screen.getByRole('status')).toHaveTextContent(/judging 13 reviews/i)
  })

  it('collapses again when the same product is clicked twice', async () => {
    const { user } = await renderShelf()
    const toggle = screen.getAllByRole('button', { name: /reviews, by aspect/i })[0]
    await user.click(toggle)
    await waitFor(() => expect(toggle).toHaveAttribute('aria-expanded', 'true'))
    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
  })

  it('offers the breakdown as a table too', async () => {
    const { user } = await renderShelf()
    await user.click(firstBreakdownToggle())
    expect(await screen.findByText('Show as a table')).toBeInTheDocument()
  })
})

describe('honesty about the data', () => {
  it('says the products are invented and the reviews are not', async () => {
    await renderShelf()
    expect(screen.getByText(/the products are invented; the reviews are not/i))
      .toBeInTheDocument()
  })

  it('admits some laptop reviews are sentences the model trained on', async () => {
    await renderShelf()
    expect(screen.getByText(/which flatters it/i)).toBeInTheDocument()
  })
})

describe('failure', () => {
  it('reports a catalogue that cannot be loaded', async () => {
    mockedGetProducts.mockRejectedValue(new ApiRequestError('model not loaded', 503))
    render(<StorePage />)
    expect(await screen.findByRole('alert')).toHaveTextContent('model not loaded')
  })

  it('still shows the shelf when only the scoring failed', async () => {
    // The scores are an enhancement; losing them must not lose the store.
    mockedGetCatalogue.mockRejectedValue(new ApiRequestError('model not loaded', 503))
    render(<StorePage />)
    await waitFor(() => expect(shelfOrder().length).toBe(PRODUCTS.length))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
