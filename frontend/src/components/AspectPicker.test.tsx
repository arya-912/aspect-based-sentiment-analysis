/**
 * `AspectPicker` -- the component whose whole job is making one class of
 * mistake unreachable.
 *
 * An aspect is in regime for this model only when it is a contiguous span of
 * the review, because every training example took its aspect from a literal
 * span of its sentence. So the invariant under test is: whatever route the
 * user takes, `onPick` receives a verbatim slice of the review.
 */

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import AspectPicker from './AspectPicker'

const REVIEW = 'It comes with a 500GB hard drive and a battery.'

function setup(text = REVIEW, selected = '') {
  const onPick = vi.fn()
  const user = userEvent.setup()
  const view = render(<AspectPicker text={text} onPick={onPick} selected={selected} />)
  return { onPick, user, view }
}

/** The word buttons, in document order. */
function wordButtons(): HTMLElement[] {
  return screen.getAllByRole('button')
}

function wordButton(label: string): HTMLElement {
  const found = wordButtons().find((button) => button.textContent === label)
  if (!found) throw new Error(`no word button "${label}"`)
  return found
}

describe('rendering the review', () => {
  it('shows every word as its own button', () => {
    setup()
    expect(wordButtons().map((b) => b.textContent)).toEqual([
      'It', 'comes', 'with', 'a', '500GB', 'hard', 'drive', 'and', 'a', 'battery',
    ])
  })

  it('renders nothing at all for text with no words', () => {
    const { view } = setup('   ')
    expect(view.container).toBeEmptyDOMElement()
  })

  it('keeps the review’s own punctuation, not a rebuilt approximation', () => {
    setup()
    // The trailing full stop and the real spacing have to survive: this is
    // meant to show the review as written.
    expect(screen.getByRole('group').textContent).toBe(REVIEW)
  })

  it('preserves punctuation and spacing exactly, including a double space', () => {
    const odd = 'The screen -- really the panel --  is dim.'
    setup(odd)
    expect(screen.getByRole('group').textContent).toBe(odd)
  })
})

describe('picking', () => {
  it('passes the clicked word verbatim', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('battery'))
    expect(onPick).toHaveBeenCalledWith('battery')
  })

  it('shift-click extends to a contiguous span', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('hard'))
    await user.keyboard('{Shift>}')
    await user.click(wordButton('drive'))
    await user.keyboard('{/Shift}')
    expect(onPick).toHaveBeenLastCalledWith('hard drive')
  })

  it('a multi-word pick is a verbatim slice of the review', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('500GB'))
    await user.keyboard('{Shift>}')
    await user.click(wordButton('drive'))
    await user.keyboard('{/Shift}')
    const picked = onPick.mock.lastCall?.[0] as string
    expect(picked).toBe('500GB hard drive')
    expect(REVIEW).toContain(picked)
  })

  it('extends backwards as well as forwards', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('drive'))
    await user.keyboard('{Shift>}')
    await user.click(wordButton('hard'))
    await user.keyboard('{/Shift}')
    expect(onPick).toHaveBeenLastCalledWith('hard drive')
  })

  it('a plain click after an extend starts a new single-word pick', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('hard'))
    await user.keyboard('{Shift>}')
    await user.click(wordButton('drive'))
    await user.keyboard('{/Shift}')
    await user.click(wordButton('battery'))
    expect(onPick).toHaveBeenLastCalledWith('battery')
  })
})

describe('the lit span follows the field, not the clicks', () => {
  it('lights the words of an aspect that was typed rather than clicked', () => {
    setup(REVIEW, 'hard drive')
    expect(wordButton('hard')).toHaveAttribute('aria-pressed', 'true')
    expect(wordButton('drive')).toHaveAttribute('aria-pressed', 'true')
    expect(wordButton('battery')).toHaveAttribute('aria-pressed', 'false')
  })

  it('matches case-insensitively, as the field may be typed in any case', () => {
    setup(REVIEW, 'HARD DRIVE')
    expect(wordButton('hard')).toHaveAttribute('aria-pressed', 'true')
  })

  it('ignores surrounding whitespace in the field', () => {
    setup(REVIEW, '  battery  ')
    expect(wordButton('battery')).toHaveAttribute('aria-pressed', 'true')
  })

  it('lights nothing for an aspect that is not a span of the review', () => {
    setup(REVIEW, 'keyboard')
    for (const button of wordButtons()) {
      expect(button).toHaveAttribute('aria-pressed', 'false')
    }
  })

  it('lights nothing when the field is empty', () => {
    setup(REVIEW, '')
    for (const button of wordButtons()) {
      expect(button).toHaveAttribute('aria-pressed', 'false')
    }
  })

  it('lights every occurrence of a word that repeats', () => {
    // "a" occurs twice; the field names one aspect and cannot say which.
    setup(REVIEW, 'a')
    const pressed = wordButtons().filter(
      (b) => b.getAttribute('aria-pressed') === 'true',
    )
    expect(pressed.map((b) => b.textContent)).toEqual(['a', 'a'])
  })
})

describe('keyboard: one tab stop for the group', () => {
  it('gives exactly one word the tab stop', () => {
    setup()
    const tabbable = wordButtons().filter((b) => b.getAttribute('tabindex') === '0')
    expect(tabbable).toHaveLength(1)
    // A 30-word review must not become 30 tab stops.
    expect(wordButtons().filter((b) => b.getAttribute('tabindex') === '-1')).toHaveLength(
      wordButtons().length - 1,
    )
  })

  it('moves focus with the arrow keys', async () => {
    const { user } = setup()
    await user.tab()
    expect(wordButton('It')).toHaveFocus()
    await user.keyboard('{ArrowRight}')
    expect(wordButton('comes')).toHaveFocus()
    await user.keyboard('{ArrowLeft}')
    expect(wordButton('It')).toHaveFocus()
  })

  it('jumps to the ends with Home and End', async () => {
    const { user } = setup()
    await user.tab()
    await user.keyboard('{End}')
    expect(wordButton('battery')).toHaveFocus()
    await user.keyboard('{Home}')
    expect(wordButton('It')).toHaveFocus()
  })

  it('does not run off either end', async () => {
    const { user } = setup()
    await user.tab()
    await user.keyboard('{ArrowLeft}{ArrowLeft}')
    expect(wordButton('It')).toHaveFocus()
    await user.keyboard('{End}{ArrowRight}')
    expect(wordButton('battery')).toHaveFocus()
  })

  it('shift-arrow extends the span, as text selection does', async () => {
    const { onPick, user } = setup()
    await user.click(wordButton('hard'))
    await user.keyboard('{Shift>}{ArrowRight}{/Shift}')
    expect(onPick).toHaveBeenLastCalledWith('hard drive')
  })

  it('a plain arrow move does not pick anything', async () => {
    const { onPick, user } = setup()
    await user.tab()
    onPick.mockClear()
    await user.keyboard('{ArrowRight}')
    expect(onPick).not.toHaveBeenCalled()
  })

  it('Enter on a focused word picks it', async () => {
    const { onPick, user } = setup()
    await user.tab()
    await user.keyboard('{Enter}')
    expect(onPick).toHaveBeenCalledWith('It')
  })
})

describe('the review being edited underneath it', () => {
  it('survives the text becoming shorter than the focused index', async () => {
    const onPick = vi.fn()
    const user = userEvent.setup()
    const { rerender } = render(
      <AspectPicker text={REVIEW} onPick={onPick} selected="" />,
    )
    await user.tab()
    await user.keyboard('{End}')

    // The indices now point past the end of the new, shorter review.
    expect(() =>
      rerender(<AspectPicker text="It comes" onPick={onPick} selected="" />),
    ).not.toThrow()
    expect(wordButtons().map((b) => b.textContent)).toEqual(['It', 'comes'])
    const tabbable = wordButtons().filter((b) => b.getAttribute('tabindex') === '0')
    expect(tabbable).toHaveLength(1)
  })
})

describe('accessibility', () => {
  it('names the group so a screen reader can explain it', () => {
    setup()
    expect(screen.getByRole('group')).toHaveAccessibleName(
      'Pick an aspect by selecting words from the review',
    )
  })

  it('reports selection with aria-pressed, not colour alone', () => {
    setup(REVIEW, 'battery')
    expect(wordButton('battery')).toHaveAttribute('aria-pressed', 'true')
  })
})
