import { useState } from 'react'

import AnalysisPage from './pages/AnalysisPage'
import PerformancePage from './pages/PerformancePage'

type Tab = 'analysis' | 'performance'

const TABS: ReadonlyArray<{ id: Tab; label: string }> = [
  { id: 'analysis', label: 'Analysis' },
  { id: 'performance', label: 'Performance' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('analysis')

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <h1 className="text-xl font-bold text-gray-900">
            Aspect-Based Sentiment Analysis
          </h1>
          <p className="mt-1 text-sm text-gray-600">Laptop reviews, aspect by aspect.</p>

          <nav className="mt-4 flex gap-2" aria-label="Sections">
            {TABS.map(({ id, label }) => {
              const isActive = activeTab === id
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setActiveTab(id)}
                  aria-current={isActive ? 'page' : undefined}
                  className={`rounded-md px-4 py-2 text-sm font-medium ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {label}
                </button>
              )
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-6">
        {/*
          Remounted per tab via `key`, so the Performance page refetches each
          time it is opened rather than showing a stale snapshot.
        */}
        {activeTab === 'analysis' ? (
          <AnalysisPage key="analysis" />
        ) : (
          <PerformancePage key="performance" />
        )}
      </main>
    </div>
  )
}
