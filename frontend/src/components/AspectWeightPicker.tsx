import { WEIGHT_STEPS } from '../lib/weights'

interface AspectWeightPickerProps {
  aspects: string[]
  /** aspect -> weight. An absent or zero weight means "ignore this aspect". */
  weights: Record<string, number>
  onChange: (weights: Record<string, number>) => void
}

/**
 * "Best for me": say which aspects matter and by how much.
 *
 * The reason this cannot be done with an overall rating -- and the reason it
 * is the most useful thing on the page. Two shoppers reading the same reviews
 * want different products, because one cares about battery and the other
 * about price. A single score answers for neither.
 *
 * Weights are small integers rather than a slider: 0-3 is the granularity
 * anyone can actually express about their own preferences, and a pixel-level
 * slider would imply a precision the underlying shares do not have.
 */
export default function AspectWeightPicker({
  aspects,
  weights,
  onChange,
}: AspectWeightPickerProps) {
  const active = aspects.filter((aspect) => (weights[aspect] ?? 0) > 0)

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium text-ink">Best for me</p>
        {active.length > 0 && (
          <button
            type="button"
            onClick={() => onChange({})}
            className="text-xs text-muted underline-offset-2 hover:text-ink-2 hover:underline"
          >
            Clear weights
          </button>
        )}
      </div>
      <p className="mt-0.5 text-xs text-muted">
        Weight the aspects you care about; the shelf reorders to match.
      </p>

      <ul className="mt-2 space-y-1.5">
        {aspects.map((aspect) => {
          const current = weights[aspect] ?? 0
          return (
            <li key={aspect} className="flex items-center justify-between gap-3">
              <span
                className={`text-sm ${current > 0 ? 'font-medium text-ink' : 'text-ink-2'}`}
              >
                {aspect}
              </span>
              {/*
                A radiogroup, because the weights are mutually exclusive per
                aspect. Rendered as buttons so the whole row is one tab stop
                rather than four.
              */}
              <div
                role="radiogroup"
                aria-label={`Weight for ${aspect}`}
                className="flex shrink-0 overflow-hidden rounded-lg border border-line"
              >
                {WEIGHT_STEPS.map((step) => (
                  <button
                    key={step}
                    type="button"
                    role="radio"
                    aria-checked={current === step}
                    aria-label={
                      step === 0 ? `Ignore ${aspect}` : `${aspect} weight ${step}`
                    }
                    onClick={() => {
                      const next = { ...weights }
                      if (step === 0) delete next[aspect]
                      else next[aspect] = step
                      onChange(next)
                    }}
                    className={`px-2 py-1 text-xs tabular transition-colors ${
                      current === step
                        ? 'bg-accent text-on-accent'
                        : 'text-ink-2 hover:bg-accent/15 hover:text-ink'
                    }`}
                  >
                    {step === 0 ? '–' : `×${step}`}
                  </button>
                ))}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
