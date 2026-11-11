import { SENTIMENTS } from '../types'

interface ConfusionMatrixProps {
  /** `matrix[gold][predicted]`, indexed in class-id order. */
  matrix: number[][]
}

/** How many ramp steps `--color-seq-*` provides. */
const STEPS = 6

/**
 * Which ramp step a share of a row falls in.
 *
 * Zero is deliberately *not* step 1: an empty cell takes the plain surface, so
 * "never happened" reads as blank rather than as a faint quantity. Above zero
 * the share is bucketed across the six steps.
 */
function stepFor(share: number): number | null {
  if (share <= 0) return null
  return Math.min(STEPS, Math.max(1, Math.ceil(share * STEPS)))
}

/**
 * Where the model's mistakes actually go.
 *
 * The backend has been sending `confusion_matrix` in every `/model` response
 * from the start -- it is listed in its `EVALUATION_KEYS` -- and the UI dropped
 * it on the floor. It is the most informative thing in the report: accuracy
 * says how often the model is wrong, this says *what it is wrong about*, and
 * for this model the answer is lopsided enough to matter (most missed neutrals
 * go to negative, which is why negative precision trails its recall badly).
 *
 * Shaded by **share of the actual-label row**, not by raw count. The classes
 * are nowhere near balanced in the gold set -- 341 positive against 128
 * negative -- so shading raw counts would just redraw the class distribution
 * and the diagonal would be the only thing visible.
 *
 * Every cell prints its count, so the color is a second channel rather than
 * the only one.
 */
export default function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  const rowTotals = SENTIMENTS.map((_, gold) =>
    SENTIMENTS.reduce((sum, __, predicted) => sum + (matrix[gold]?.[predicted] ?? 0), 0),
  )

  return (
    <figure className="overflow-x-auto">
      {/* `w-auto`, not `w-full`: a 3x3 matrix stretched across a wide card turns
          cells into slabs and buries the pattern in whitespace. */}
      <table className="w-auto border-separate border-spacing-0.5 text-sm">
        <thead>
          <tr>
            <td className="p-1" />
            <th
              scope="colgroup"
              colSpan={SENTIMENTS.length}
              className="p-1 pb-2 text-center text-xs font-medium uppercase tracking-wide text-muted"
            >
              Predicted
            </th>
            <td className="p-1" />
          </tr>
          <tr>
            <td className="p-1" />
            {SENTIMENTS.map((label) => (
              <th
                key={label}
                scope="col"
                className="p-1 pb-1.5 text-center text-xs font-medium capitalize text-ink-2"
              >
                {label}
              </th>
            ))}
            <th scope="col" className="p-1 pb-1.5 text-right text-xs font-medium text-muted">
              Total
            </th>
          </tr>
        </thead>

        <tbody>
          {SENTIMENTS.map((goldLabel, gold) => {
            const total = rowTotals[gold] ?? 0
            return (
              <tr key={goldLabel}>
                <th
                  scope="row"
                  className="whitespace-nowrap py-1 pr-2 text-right text-xs font-medium capitalize text-ink-2"
                >
                  {goldLabel}
                </th>

                {SENTIMENTS.map((predictedLabel, predicted) => {
                  const count = matrix[gold]?.[predicted] ?? 0
                  const share = total > 0 ? count / total : 0
                  const step = stepFor(share)
                  const correct = gold === predicted

                  return (
                    <td
                      key={predictedLabel}
                      title={`${count} ${goldLabel} example${count === 1 ? '' : 's'} predicted ${predictedLabel} — ${(share * 100).toFixed(1)}% of all ${goldLabel}`}
                      className={`tabular w-20 rounded-[4px] p-2 text-center font-medium ${
                        // A ring, not a different hue: the diagonal is already
                        // meaningful by position, and recoloring it would break
                        // the one-hue sequential scale.
                        correct ? 'ring-1 ring-inset ring-ink/40' : ''
                      }`}
                      style={
                        step === null
                          ? undefined
                          : {
                              backgroundColor: `var(--color-seq-${step})`,
                              color: `var(--color-seq-${step}-ink)`,
                            }
                      }
                    >
                      {count}
                    </td>
                  )
                })}

                <td className="tabular py-1 pl-2 text-right text-xs text-muted">{total}</td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {/*
        A <figcaption> rather than a <caption>: a table caption is laid out to
        the table's width, and this table is deliberately narrow, which wrapped
        the explanation into a hard-to-read column.
      */}
      <figcaption className="mt-3 max-w-prose text-xs text-muted">
        Rows are the true label, columns the model&rsquo;s prediction; the diagonal is
        correct. Shading is each cell&rsquo;s share of its own row, so rows of very
        different sizes stay comparable.
      </figcaption>
    </figure>
  )
}
