/**
 * Number formatting, in one place.
 *
 * Three components previously carried their own copy of the percent helper,
 * each with slightly different handling of the absent case.
 */

/** A 0-1 ratio as a percentage, or an em dash when there is no number. */
export function asPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** An integer with thousands separators, or an em dash. */
export function asCount(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toLocaleString('en-US')
}
