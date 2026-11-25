import type { ReactNode } from 'react'

interface SectionHeaderProps {
  title: string
  /** One line on what this section is for, or what its numbers mean. */
  description?: ReactNode
  /** Controls or a legend, set opposite the title. */
  aside?: ReactNode
  /** `h2` for a page's own heading, `h3` for a block inside one. */
  as?: 'h2' | 'h3'
}

/**
 * The heading row every section uses.
 *
 * Extracted because the four pages had grown four slightly different versions
 * of the same flex row, and a dashboard reads as one product mostly through
 * the parts that repeat. The aside slot keeps a legend or a control on the
 * heading's baseline instead of floating above the content it describes.
 */
export default function SectionHeader({
  title,
  description,
  aside,
  as = 'h3',
}: SectionHeaderProps) {
  const Heading = as

  return (
    <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
      <div className="min-w-0">
        <Heading
          className={
            as === 'h2'
              ? 'text-base font-semibold text-ink'
              : 'text-sm font-semibold text-ink'
          }
        >
          {title}
        </Heading>
        {description && (
          <p className="mt-1 max-w-prose text-sm text-ink-2">{description}</p>
        )}
      </div>
      {aside && <div className="shrink-0">{aside}</div>}
    </div>
  )
}
