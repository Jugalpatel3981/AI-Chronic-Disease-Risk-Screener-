'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Header, buttonPrimary, buttonSecondary } from '@/components/Header'
import { clearResults, loadResults, nextStep, toRows, type RowLevel, type StoredResults } from '@/lib/results'

const BADGE: Record<RowLevel, { label: string; className: string; icon: string }> = {
  low: { label: 'Low risk', className: 'bg-low-soft text-low', icon: 'M5 12l5 5 9-10' },
  moderate: { label: 'Moderate', className: 'bg-moderate-soft text-moderate', icon: 'M12 7v6M12 17h.01' },
  high: { label: 'High risk', className: 'bg-high-soft text-high', icon: 'M12 7v6M12 17h.01' },
  diagnosed: { label: 'Diagnosed', className: 'bg-line-soft text-ink-2', icon: 'M5 12h14' },
  'not-checked': { label: 'Not checked', className: 'bg-line-soft text-muted', icon: 'M5 12h14' },
}

export default function Results() {
  const router = useRouter()
  const [results, setResults] = useState<StoredResults | null | undefined>(undefined)

  useEffect(() => setResults(loadResults()), [])

  if (results === undefined) return null

  if (results === null) {
    return (
      <div className="min-h-screen">
        <Header />
        <main className="mx-auto flex max-w-[760px] flex-col items-start gap-5 px-6 py-16">
          <h1 className="font-display text-3xl font-bold">No results yet</h1>
          <p className="text-ink-2">Complete the screening to see your results here.</p>
          <Link href="/screen" className={buttonPrimary}>Start screening</Link>
        </main>
      </div>
    )
  }

  const rows = toRows(results)

  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto flex max-w-[760px] flex-col gap-6 px-6 pt-8 pb-16">
        <div className="flex flex-col gap-1">
          <h1 className="font-display text-4xl font-bold tracking-tight">Your results</h1>
          <span className="text-sm text-muted">
            {new Date(results.createdAt).toLocaleDateString(undefined, { day: 'numeric', month: 'long', year: 'numeric' })}
          </span>
        </div>

        <section aria-label="Results" className="rounded-[20px] border border-line bg-surface px-6 py-2">
          {rows.map((r, i) => {
            const badge = BADGE[r.level]
            return (
              <div key={r.key}
                className={`flex flex-wrap items-center gap-x-4 gap-y-2 py-5 ${i < rows.length - 1 ? 'border-b border-line-soft' : ''}`}>
                <div className="flex min-w-[220px] flex-1 flex-col gap-1">
                  <span className="text-[17px] font-semibold">{r.name}</span>
                  <span className="text-sm text-muted">{r.note}</span>
                </div>
                <span className={`min-w-24 text-right font-display text-2xl font-bold ${r.level === 'not-checked' ? 'text-muted' : ''}`}>
                  {r.value}
                </span>
                <span className={`flex min-w-[112px] items-center justify-center gap-1.5 rounded-full px-3 py-1.5 text-[13px] font-semibold ${badge.className}`}>
                  <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d={badge.icon} /></svg>
                  {badge.label}
                </span>
              </div>
            )
          })}
        </section>

        <section className="flex flex-col gap-1.5 rounded-2xl bg-brand-soft px-6 py-5">
          <h2 className="text-[17px] font-semibold text-brand-ink">Next step</h2>
          <p className="text-[15px] leading-relaxed text-brand-ink">{nextStep(results)}</p>
        </section>

        <div className="no-print flex flex-wrap gap-3">
          <button type="button" onClick={() => window.print()} className={buttonPrimary}>Download PDF</button>
          <button type="button" className={buttonSecondary} onClick={() => { clearResults(); router.push('/screen') }}>
            Start over
          </button>
        </div>
      </main>
    </div>
  )
}
