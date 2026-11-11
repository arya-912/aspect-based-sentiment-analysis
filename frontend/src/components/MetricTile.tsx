import Card from './Card'

interface MetricTileProps {
  label: string
  value: string
  /** Optional smaller line beneath, e.g. what the number was measured on. */
  detail?: string
  /** `true` for the headline metric, which gets the larger figure. */
  emphasis?: boolean
}

/**
 * One number, stated plainly.
 *
 * The figures stay proportional rather than tabular: tabular digits are for
 * columns that have to align vertically, and a standalone number set in them
 * just looks loosely spaced.
 */
export default function MetricTile({ label, value, detail, emphasis = false }: MetricTileProps) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p
        className={`mt-1 font-semibold text-ink ${emphasis ? 'text-3xl' : 'text-2xl'}`}
      >
        {value}
      </p>
      {detail !== undefined && <p className="mt-1 text-xs text-muted">{detail}</p>}
    </Card>
  )
}
