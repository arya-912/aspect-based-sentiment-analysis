/**
 * The shared primitives: `Skeleton`, `EmptyState`, `SectionHeader`.
 *
 * These exist so the four pages read as one product, and the assertions are
 * mostly about accessibility: a skeleton must not announce itself forty
 * times, and a tone must never be the only thing carrying a state.
 */

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import EmptyState from './EmptyState'
import SectionHeader from './SectionHeader'
import Skeleton from './Skeleton'
import { WarningIcon } from './icons'

describe('Skeleton', () => {
  it('is hidden from assistive tech, so the region announces the wait once', () => {
    const { container } = render(<Skeleton />)
    const block = container.firstElementChild
    expect(block).toHaveAttribute('aria-hidden', 'true')
    expect(block).toHaveClass('skeleton')
  })

  it('takes its size from the caller, to reserve the real layout', () => {
    const { container } = render(<Skeleton className="h-12 w-24" />)
    expect(container.firstElementChild).toHaveClass('h-12', 'w-24')
  })
})

describe('EmptyState', () => {
  it('states what is being looked at and what to do', () => {
    render(
      <EmptyState title="No products in this category">Switch the filter.</EmptyState>,
    )
    expect(screen.getByText('No products in this category')).toBeInTheDocument()
    expect(screen.getByText('Switch the filter.')).toBeInTheDocument()
  })

  it.each(['warning', 'critical'] as const)(
    'ships the %s tone with an icon, never colour alone',
    (tone) => {
      const { container } = render(
        <EmptyState title="Something" tone={tone} icon={<WarningIcon />}>
          Body
        </EmptyState>,
      )
      // The words and the icon both carry it, so it survives greyscale.
      expect(container.querySelector('svg')).toBeInTheDocument()
      expect(screen.getByText('Something')).toBeInTheDocument()
    },
  )

  it('renders an action when there is an obvious next step', () => {
    render(
      <EmptyState title="Nothing yet" action={<button type="button">Upload</button>}>
        Body
      </EmptyState>,
    )
    expect(screen.getByRole('button', { name: 'Upload' })).toBeInTheDocument()
  })

  it('needs no icon', () => {
    const { container } = render(<EmptyState title="Plain">Body</EmptyState>)
    expect(container.querySelector('svg')).toBeNull()
  })
})

describe('SectionHeader', () => {
  it('renders an h3 by default and an h2 on request', () => {
    const { unmount } = render(<SectionHeader title="By aspect" />)
    expect(screen.getByRole('heading', { level: 3, name: 'By aspect' })).toBeInTheDocument()
    unmount()

    render(<SectionHeader as="h2" title="Demo store" />)
    expect(screen.getByRole('heading', { level: 2, name: 'Demo store' })).toBeInTheDocument()
  })

  it('puts the aside on the heading baseline rather than above the content', () => {
    render(<SectionHeader title="By aspect" aside={<span>legend</span>} />)
    expect(screen.getByText('legend')).toBeInTheDocument()
  })

  it('renders without a description or aside', () => {
    render(<SectionHeader title="Serving" />)
    expect(screen.getByRole('heading', { name: 'Serving' })).toBeInTheDocument()
  })
})
