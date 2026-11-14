import { useEffect, useState } from 'react'

import HealthBanner from './components/HealthBanner'
import ThemeToggle from './components/ThemeToggle'
import { domainFor } from './lib/domains'
import AnalysisPage from './pages/AnalysisPage'
import PerformancePage from './pages/PerformancePage'
import { getModelMeta } from './services/api'

type Tab = 'analysis' | 'performance'

const TABS: ReadonlyArray<{ id: Tab; label: string }> = [
  { id: 'analysis', label: 'Analysis' },
  { id: 'performance', label: 'Performance' },
]

function tabFromHash(): Tab {
  return window.location.hash === '#performance' ? 'performance' : 'analysis'
}

/**
 * Which section is open lives in the URL fragment rather than in component
 * state, so the browser's back button works, a reload stays where it was, and
 * the performance view can be linked to directly. The sections are real
 * navigation, so they are anchors -- not buttons dressed as tabs.
 */
function useHashTab(): Tab {
  const [tab, setTab] = useState<Tab>(tabFromHash)

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return tab
}

/**
 * The corpus the served artifact was trained on, or null while unknown.
 *
 * Asked once, because the answer changes only when the backend is restarted
 * against a different artifact. A failure is not surfaced: this only decides
 * wording and which examples to offer, and `HealthBanner` already reports a
 * backend that cannot answer. Falling back to generic copy is correct, not an
 * error state.
 */
function useServedCorpus(): string | null {
  const [corpus, setCorpus] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    getModelMeta(controller.signal)
      .then((meta) => setCorpus(meta.corpus ?? null))
      .catch(() => undefined)
    return () => controller.abort()
  }, [])

  return corpus
}

export default function App() {
  const activeTab = useHashTab()
  const domain = domainFor(useServedCorpus())

  return (
    <div className="min-h-screen bg-plane">
      {/*
        Sticky, because the Analysis page is a scrolling list of results and
        losing the section switcher partway down it is worse than the 60px the
        header costs.
      */}
      <header className="sticky top-0 z-10 border-b border-line bg-plane/85 backdrop-blur">
        <div className="mx-auto max-w-5xl px-4">
          <div className="flex items-center justify-between gap-4 py-3">
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold text-ink">
                Aspect-Based Sentiment Analysis
              </h1>
              {/*
                Follows the served artifact. Until /model answers -- and for a
                corpus domains.ts does not know -- it says what is true of
                every artifact rather than guessing a subject.
              */}
              <p className="truncate text-xs text-muted">
                {domain ? `${domain.subject}, one aspect at a time` : 'One aspect at a time'}
              </p>
            </div>
            <ThemeToggle />
          </div>

          <nav className="-mb-px flex gap-1" aria-label="Sections">
            {TABS.map(({ id, label }) => {
              const isActive = activeTab === id
              return (
                <a
                  key={id}
                  href={`#${id}`}
                  aria-current={isActive ? 'page' : undefined}
                  className={`border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'border-accent text-ink'
                      : 'border-transparent text-muted hover:border-line hover:text-ink-2'
                  }`}
                >
                  {label}
                </a>
              )
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {/*
          Outside the tab switch, and above it: it reports whether the backend
          can serve anything at all, which is true of both tabs and is the
          first thing worth knowing when either one looks broken.
        */}
        <HealthBanner />

        {/*
          Remounted per tab via `key`, so the Performance page refetches each
          time it is opened rather than showing a stale snapshot.
        */}
        {activeTab === 'analysis' ? (
          <AnalysisPage key="analysis" examples={domain?.examples} />
        ) : (
          <PerformancePage key="performance" />
        )}
      </main>

      <footer className="mx-auto max-w-5xl px-4 pb-8 pt-2">
        <p className="text-xs text-muted">
          BERT with aspect attention
          {domain ? `, fine-tuned on ${domain.corpusName}` : ''}. Predictions are about
          the aspect term you give, not the sentence as a whole.
        </p>
      </footer>
    </div>
  )
}
