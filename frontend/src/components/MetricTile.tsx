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
 * The figure is set in the mono face. That is the type pairing doing its job
 * rather than decoration: these three tiles sit side by side and are read as
 * a set, so their digits need the same width or the eye reads a difference in
 * length as a difference in size.
 *
 * The figures stay *proportional* within that face, not tabular: tabular
 * digits are for columns that align vertically, and a standalone number set
 * in them looks loosely spaced.
 *
 * The emphasised tile gets a heavier rule above it rather than a different
 * background, so the set still reads as one row of the same kind of thing.
 */
export default function MetricTile({ label, value, detail, emphasis = false }: MetricTileProps) {
  return (
    <Card className={`p-4 ${emphasis ? 'border-t-2 border-t-accent' : ''}`}>
      <p className="text-xs font-medium text-muted">{label}</p>
      <p
        className={`mt-1.5 font-mono font-semibold tracking-tight text-ink ${
          emphasis ? 'text-3xl' : 'text-2xl'
        }`}
      >
        {value}
      </p>
      {detail !== undefined && (
        <p className="mt-1.5 text-xs/relaxed text-muted">{detail}</p>
      )}
    </Card>
  )
}
