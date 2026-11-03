import Link from 'next/link'

export function Logo() {
  return (
    <Link href="/" className="inline-flex items-center gap-2.5 text-ink no-underline">
      <svg aria-hidden="true" width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="#0b6b61"
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="24" height="24" rx="7" />
        <path d="M7 15h4l2-5 3 9 2-4h3" />
      </svg>
      <span className="font-display text-lg font-bold">Risk Screener</span>
    </Link>
  )
}

export function Header({ right }: { right?: React.ReactNode }) {
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-[1160px] flex-wrap items-center justify-between gap-4 px-6 py-4">
        <Logo />
        {right && <div className="text-sm text-muted">{right}</div>}
      </div>
    </header>
  )
}

export const buttonPrimary =
  'inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-7 text-[15px] font-semibold text-white ' +
  'transition-colors hover:bg-brand-dark disabled:cursor-not-allowed disabled:opacity-60'

export const buttonSecondary =
  'inline-flex min-h-12 items-center justify-center rounded-xl border border-line-strong bg-surface px-5 ' +
  'text-[15px] font-semibold text-ink transition-colors hover:bg-line-soft'
