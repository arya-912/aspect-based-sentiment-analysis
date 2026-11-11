import { asCount, asPercent } from '../lib/format'
import { SENTIMENTS, type PerClassMetrics, type Sentiment } from '../types'
import BarRow from './BarRow'

interface PerClassTableProps {
  perClass: Record<string, PerClassMetrics>
}

/** Literal class names, as everywhere else in this UI. */
const BARS: Record<Sentiment, string> = {
  negative: 'bg-neg',
  neutral: 'bg-neu',
  positive: 'bg-pos',
}

/**
 * Per-class scores, as bars and then as a table.
 *
 * Both, not one: the bars make the gap between classes obvious at a glance --
 * and the gap is the story, since `neutral` trails the other two badly -- while
 * the table is the exact-numbers view, which is also the relief channel that
 * lets the bars be colored at all.
 *
 * Each bar takes its own class's color, following the entity rather than the
 * value. Rows iterate a fixed label order rather than `Object.keys(perClass)`,
 * so the order does not depend on JSON key order and a label the report omits
 * still shows a row with dashes.
 */
export default function PerClassTable({ perClass }: PerClassTableProps) {
  return (
    <div className="space-y-5">
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
          F1 by class
        </p>
        {/* Capped: three bars do not get more readable by being 1200px long. */}
        <div className="max-w-md space-y-1.5">
          {SENTIMENTS.map((label) => (
            <BarRow
              key={label}
              label={label}
              value={asPercent(perClass[label]?.f1)}
              fraction={perClass[label]?.f1 ?? 0}
              barClassName={BARS[label]}
              labelClassName="w-20 capitalize"
            />
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[26rem] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th scope="col" className="py-2 pr-2 text-xs font-medium uppercase tracking-wide text-muted">
                Label
              </th>
              <th scope="col" className="py-2 px-2 text-right text-xs font-medium uppercase tracking-wide text-muted">
                F1
              </th>
              <th scope="col" className="py-2 px-2 text-right text-xs font-medium uppercase tracking-wide text-muted">
                Precision
              </th>
              <th scope="col" className="py-2 px-2 text-right text-xs font-medium uppercase tracking-wide text-muted">
                Recall
              </th>
              <th scope="col" className="py-2 pl-2 text-right text-xs font-medium uppercase tracking-wide text-muted">
                Support
              </th>
            </tr>
          </thead>
          <tbody>
            {SENTIMENTS.map((label) => {
              const scores = perClass[label]
              return (
                <tr key={label} className="border-b border-line last:border-b-0">
                  <th scope="row" className="py-2 pr-2 text-left font-medium capitalize text-ink">
                    <span className="inline-flex items-center gap-2">
                      <span className={`h-2 w-2 rounded-full ${BARS[label]}`} aria-hidden="true" />
                      {label}
                    </span>
                  </th>
                  <td className="py-2 px-2 text-right font-medium text-ink">
                    {asPercent(scores?.f1)}
                  </td>
                  <td className="py-2 px-2 text-right text-ink-2">{asPercent(scores?.precision)}</td>
                  <td className="py-2 px-2 text-right text-ink-2">{asPercent(scores?.recall)}</td>
                  <td className="py-2 pl-2 text-right text-ink-2">{asCount(scores?.support)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
