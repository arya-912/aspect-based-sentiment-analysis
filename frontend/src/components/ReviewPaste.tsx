import { useId, useRef, useState } from 'react'

interface ReviewPasteProps {
  value: string
  onChange: (value: string) => void
  onAnalyze: () => void
  busy: boolean
  /** Fills the box with a worked example. */
  onLoadExample?: () => void
  exampleLabel?: string
}

/** Characters we will read out of an uploaded file, matching the API's ceiling. */
const MAX_UPLOAD_CHARS = 200 * 5000

/**
 * Where a review set comes in: pasted, or dropped in as a file.
 *
 * One review per line. That is a real constraint rather than a convenience --
 * the backend splits each review into sentences and reports which review each
 * example came from, so it has to know where one review ends. A blank line is
 * ignored, so a file with paragraph spacing works too.
 *
 * The file input accepts `.txt` and `.csv`; a CSV is read as lines, not
 * parsed, because a one-column export is the common shape and a real parser
 * here would be guessing which column holds the review.
 */
export default function ReviewPaste({
  value,
  onChange,
  onAnalyze,
  busy,
  onLoadExample,
  exampleLabel,
}: ReviewPasteProps) {
  const textareaId = useId()
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)

  const lineCount = value.split('\n').filter((line) => line.trim() !== '').length

  async function handleFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setFileError(null)
    try {
      const text = await file.text()
      if (text.length > MAX_UPLOAD_CHARS) {
        setFileError('That file is too large; paste a sample of it instead.')
        return
      }
      onChange(text)
    } catch {
      setFileError('Could not read that file.')
    } finally {
      // Cleared so choosing the same file twice fires change again.
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <label htmlFor={textareaId} className="text-sm font-medium text-ink">
          Reviews, one per line
        </label>
        <span className="text-xs text-muted">
          {lineCount === 1 ? '1 review' : `${lineCount} reviews`}
        </span>
      </div>

      <textarea
        id={textareaId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        rows={8}
        spellCheck={false}
        placeholder={'Battery life is amazing but the screen is dim.\nThe keyboard feels great, but the trackpad is unresponsive.'}
        className="mt-1.5 w-full resize-y rounded-lg border border-line bg-surface p-3 font-mono text-[13px]/relaxed text-ink placeholder:text-muted focus:border-accent focus:outline-none"
      />

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onAnalyze}
          disabled={busy || lineCount === 0}
          className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-on-accent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? 'Analyzing…' : 'Analyze by aspect'}
        </button>

        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={busy}
          className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium text-ink-2 transition-colors hover:border-accent hover:text-ink disabled:opacity-50"
        >
          Upload a file
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".txt,.csv,text/plain,text/csv"
          onChange={handleFile}
          className="sr-only"
          aria-label="Upload a file of reviews, one per line"
        />

        {onLoadExample && (
          <button
            type="button"
            onClick={onLoadExample}
            disabled={busy}
            className="text-sm text-accent underline-offset-2 hover:underline disabled:opacity-50"
          >
            {exampleLabel ?? 'Load an example set'}
          </button>
        )}

        {value !== '' && (
          <button
            type="button"
            onClick={() => onChange('')}
            disabled={busy}
            className="text-sm text-muted underline-offset-2 hover:text-ink-2 hover:underline disabled:opacity-50"
          >
            Clear
          </button>
        )}
      </div>

      {fileError && (
        <p role="alert" className="mt-2 text-sm text-critical">
          {fileError}
        </p>
      )}
    </div>
  )
}
