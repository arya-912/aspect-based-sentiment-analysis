import { asCount, asPercent } from '../lib/format'
import { SENTIMENTS, type AspectSummary } from '../types'

interface AspectTableProps {
  aspects: AspectSummary[]
}

/**
 * The share bars as a table.
 *
 * Required rather than optional: a chart whose values exist only as bar widths
 * and hover text is unreadable to a screen reader and unusable for copying a
 * figure out. Every number the bars encode is here in a form that can be read
 * in order, sorted by eye and pasted into a report.
 */
export default function AspectTable({ aspects }: AspectTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[30rem] text-sm">
        <caption className="sr-only">
          Sentence counts and shares per aspect, per sentiment label
        </caption>
        <thead>
          <tr className="border-b border-line text-left">
            <th scope="col" className="py-2 pr-2 text-xs font-medium uppercase tracking-wide text-muted">
              Aspect
            </th>
            {SENTIMENTS.map((label) => (
              <th
                key={label}
                scope="col"
                className="px-2 py-2 text-right text-xs font-medium uppercase tracking-wide text-muted"
              >
                {label}
              </th>
            ))}
            <th scope="col" className="py-2 pl-2 text-right text-xs font-medium uppercase tracking-wide text-muted">
              Sentences
            </th>
          </tr>
        </thead>
        <tbody>
          {aspects.map((summary) => (
            <tr key={summary.aspect} className="border-b border-line last:border-b-0">
              <th scope="row" className="py-2 pr-2 text-left font-medium text-ink">
                {summary.aspect}
              </th>
              {SENTIMENTS.map((label) => (
                <td key={label} className="tabular px-2 py-2 text-right text-ink-2">
                  {summary.counts[label] ? (
                    <>
                      {asPercent(summary.shares[label] ?? 0, 0)}
                      <span className="text-muted"> ({summary.counts[label]})</span>
                    </>
                  ) : (
                    '—'
                  )}
                </td>
              ))}
              <td className="tabular py-2 pl-2 text-right text-ink-2">
                {asCount(summary.total)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
