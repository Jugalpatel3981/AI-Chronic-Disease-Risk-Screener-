import Link from 'next/link'
import { Logo, buttonPrimary } from '@/components/Header'

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto w-full max-w-[1160px] px-6 py-5">
        <Logo />
      </header>

      <main className="mx-auto flex w-full max-w-[760px] flex-1 flex-col items-center gap-7 px-6 py-24 text-center">
        <h1 className="font-display text-4xl leading-tight font-bold tracking-tight text-balance sm:text-[52px]">
          Check your health risks in 5 minutes
        </h1>
        <p className="text-lg leading-relaxed text-pretty text-ink-2 sm:text-[19px]">
          Answer a few questions and add a nail photo and a retina photo. AI screens for hypertension,
          diabetes, kidney disease, anemia and diabetic retinopathy.
        </p>
        <Link href="/screen" className={`${buttonPrimary} min-h-14 px-9 text-[17px]`}>
          Start screening
        </Link>
        <span className="text-sm text-muted">Free · No sign-up · Photos are never stored</span>
      </main>

      <footer className="mx-auto w-full max-w-[1160px] px-6 py-6 text-center text-[13px] text-muted">
        A screening tool for education and research — not a medical diagnosis.
      </footer>
    </div>
  )
}
