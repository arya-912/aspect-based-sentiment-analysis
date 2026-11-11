import { useRef, useState } from 'react'

import { sliceSpan, splitWords } from '../lib/words'

interface AspectPickerProps {
  /** The review to pick words out of. */
  text: string
  /** Called with the verbatim slice covering the selected words. */
  onPick: (aspect: string) => void
  /** The aspect currently in the field, so an existing pick stays lit. */
  selected: string
}

/**
 * Pick the aspect by clicking words in the review.
 *
 * Why this exists rather than only a text field: an aspect is meaningful to
 * this model *only* when it is a contiguous span of the review. Every training
 * example took its aspect from a literal span of its sentence, so a typed
 * aspect that is absent -- or present only as scattered words -- is outside the
 * regime the model learned, and it will answer just as confidently anyway.
 * Building the aspect by slicing the review's own characters makes that class
 * of mistake unreachable instead of merely warning about it afterwards.
 *
 * Click selects a word; shift-click (or shift-arrow) extends to a contiguous
 * run, which is exactly the shape the model accepts.
 *
 * Keyboard handling is the composite-widget pattern -- one tab stop for the
 * whole group, arrows to move within it -- because a 30-word review would
 * otherwise be 30 tab stops between the textarea and the submit button.
 */
export default function AspectPicker({ text, onPick, selected }: AspectPickerProps) {
  const words = splitWords(text)

  /** Where a shift-extend measures from. */
  const [anchorState, setAnchor] = useState(0)
  /** Which word owns the group's single tab stop. */
  const [focusState, setFocusIndex] = useState(0)
  const buttons = useRef<Array<HTMLButtonElement | null>>([])

  if (words.length === 0) return null

  // Editing the review shorter can leave either index past the end. Clamped on
  // read rather than corrected in an effect: an effect would have to re-render
  // to fix state that is only ever wrong in between, and the right value is
  // already derivable here.
  const lastIndex = words.length - 1
  const anchor = Math.min(anchorState, lastIndex)
  const focusIndex = Math.min(focusState, lastIndex)

  /**
   * Which words are lit.
   *
   * Derived from the aspect text rather than held as state, so the highlight
   * follows the field even when it was typed or came from an example -- one
   * source of truth, and no way for the two to disagree.
   */
  const needle = selected.trim().toLowerCase()
  const lit = new Set<number>()
  if (needle !== '') {
    for (let start = 0; start < words.length; start += 1) {
      for (let end = start; end < words.length; end += 1) {
        if (sliceSpan(text, words, start, end).toLowerCase() === needle) {
          for (let i = start; i <= end; i += 1) lit.add(i)
        }
      }
    }
  }

  function pick(from: number, to: number) {
    onPick(sliceSpan(text, words, from, to))
  }

  function handleClick(index: number, extend: boolean) {
    if (extend) {
      pick(anchor, index)
    } else {
      setAnchor(index)
      pick(index, index)
    }
    setFocusIndex(index)
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    const moves: Record<string, number> = {
      ArrowLeft: index - 1,
      ArrowRight: index + 1,
      Home: 0,
      End: words.length - 1,
    }
    const next = moves[event.key]
    if (next === undefined) return

    event.preventDefault()
    const clamped = Math.max(0, Math.min(words.length - 1, next))
    setFocusIndex(clamped)
    buttons.current[clamped]?.focus()
    // Shift turns the move into an extend, matching how text selection behaves.
    if (event.shiftKey) pick(anchor, clamped)
  }

  return (
    <div>
      {/*
        Inline flow, not flexbox: the gaps between words are real text, and a
        flex container drops whitespace-only children outright, which would
        reflow the sentence into evenly-spaced chips. Rendering the gaps also
        keeps the punctuation -- this has to show the review as written, not a
        rebuilt approximation of it.
      */}
      <div
        className="whitespace-pre-wrap rounded-lg border border-line bg-plane p-2.5 text-[15px]/relaxed"
        role="group"
        aria-label="Pick an aspect by selecting words from the review"
      >
        {words.map((word, index) => {
          const previous = words[index - 1]
          const gap = previous === undefined
            ? text.slice(0, word.start)
            : text.slice(previous.end, word.start)
          // A gap flanked by two selected words is inside the span, so it gets
          // the same fill -- otherwise a two-word aspect reads as two aspects.
          const gapLit = index > 0 && lit.has(index - 1) && lit.has(index)

          return (
            <span key={`${word.start}-${word.text}`}>
              {gap !== '' && (
                <span className={gapLit ? 'bg-accent text-on-accent' : undefined}>{gap}</span>
              )}
              <button
                ref={(element) => {
                  buttons.current[index] = element
                }}
                type="button"
                tabIndex={index === focusIndex ? 0 : -1}
                aria-pressed={lit.has(index)}
                onClick={(event) => handleClick(index, event.shiftKey)}
                onKeyDown={(event) => handleKeyDown(event, index)}
                className={`transition-colors ${
                  lit.has(index)
                    ? 'bg-accent text-on-accent'
                    : 'rounded text-ink-2 hover:bg-accent/15 hover:text-ink'
                }`}
              >
                {word.text}
              </button>
            </span>
          )
        })}
        {/* Trailing punctuation after the last word, e.g. the full stop. */}
        {(() => {
          const last = words[words.length - 1]
          return last === undefined ? null : <span>{text.slice(last.end)}</span>
        })()}
      </div>
      <p className="mt-1.5 text-xs text-muted">
        Click a word to set the aspect, shift-click a second to span several. A span
        taken from the review is always one the model can judge.
      </p>
    </div>
  )
}
