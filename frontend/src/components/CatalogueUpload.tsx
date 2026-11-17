import { useId, useRef, useState } from 'react'

import { SAMPLE_CSV_URL, isApiError, resetCatalogue, uploadCatalogue } from '../services/api'
import type { CatalogueStatus, CatalogueUploadResult } from '../types'
import { WarningIcon } from './icons'

interface CatalogueUploadProps {
  status: CatalogueStatus | null
  /** Called after the active catalogue changes, so the page can refetch. */
  onChanged: (status: CatalogueStatus) => void
}

const CATEGORIES = [
  { value: 'restaurant', label: 'Restaurants' },
  { value: 'laptop', label: 'Laptops' },
] as const

/**
 * Point the store at a real review export.
 *
 * The category selector is not decoration: a file with no `category` column
 * -- which is most real exports, since they cover one domain -- cannot be
 * parsed without being told which one, and the backend rejects it rather
 * than guessing. Choosing here is cheaper than making someone edit a CSV.
 *
 * The 422 message is surfaced verbatim. The backend names the missing
 * column, the offending row or the cap that was hit, and rewording that into
 * "upload failed" would throw away the only part the user can act on.
 */
export default function CatalogueUpload({ status, onChanged }: CatalogueUploadProps) {
  const fileId = useId()
  const categoryId = useId()
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [category, setCategory] = useState<string>('restaurant')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CatalogueUploadResult | null>(null)

  const onUpload = async (file: File) => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const uploaded = await uploadCatalogue(file, category)
      setResult(uploaded)
      onChanged(uploaded)
    } catch (caught: unknown) {
      setError(isApiError(caught) ? caught.message : 'Could not read that file.')
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const onReset = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      onChanged(await resetCatalogue())
    } catch (caught: unknown) {
      setError(isApiError(caught) ? caught.message : 'Could not restore the demo catalogue.')
    } finally {
      setBusy(false)
    }
  }

  const uploaded = status?.source === 'upload'

  return (
    <div className="rounded-lg border border-line bg-plane p-3">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor={categoryId} className="block text-xs text-muted">
            These reviews are about
          </label>
          <select
            id={categoryId}
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            disabled={busy}
            className="mt-1 rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink focus:border-accent focus:outline-none disabled:opacity-50"
          >
            {CATEGORIES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={busy}
            className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-on-accent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? 'Reading…' : 'Upload catalogue'}
          </button>
          <input
            ref={fileRef}
            id={fileId}
            type="file"
            accept=".csv,text/csv"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void onUpload(file)
            }}
            className="sr-only"
            aria-label="Upload a catalogue CSV"
          />
        </div>

        <a
          href={SAMPLE_CSV_URL}
          download="absa-catalogue-sample.csv"
          className="text-sm text-accent underline-offset-2 hover:underline"
        >
          Sample CSV
        </a>

        {uploaded && (
          <button
            type="button"
            onClick={() => void onReset()}
            disabled={busy}
            className="text-sm text-muted underline-offset-2 hover:text-ink-2 hover:underline disabled:opacity-50"
          >
            Back to the demo catalogue
          </button>
        )}
      </div>

      <p className="mt-2 text-xs text-muted">
        One row per review. Columns: <code className="font-mono">product_name</code>,{' '}
        <code className="font-mono">review_text</code> (required),{' '}
        <code className="font-mono">category</code>,{' '}
        <code className="font-mono">price</code>,{' '}
        <code className="font-mono">rating</code>. Common export headers like{' '}
        <code className="font-mono">name</code> and <code className="font-mono">text</code>{' '}
        are accepted too.
      </p>

      {error && (
        <p role="alert" className="mt-2 flex gap-2 text-sm text-critical">
          <span className="mt-px shrink-0">
            <WarningIcon />
          </span>
          <span>{error}</span>
        </p>
      )}

      {result && (
        <div role="status" className="mt-2 rounded-lg border border-line bg-surface p-2.5">
          <p className="text-sm text-ink">
            Loaded <span className="font-medium">{result.source_name}</span>:{' '}
            {result.products} {result.products === 1 ? 'product' : 'products'}
            {result.products_found > result.products && <> of {result.products_found.toLocaleString('en-US')}</>}
            , from {result.rows_used.toLocaleString('en-US')} review rows.
          </p>
          {result.warnings.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {result.warnings.map((warning) => (
                <li key={warning} className="text-xs text-muted">
                  · {warning}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
