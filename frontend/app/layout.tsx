import type { Metadata } from 'next'
import { Inter, Inter_Tight } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const interTight = Inter_Tight({ subsets: ['latin'], weight: ['600', '700'], variable: '--font-inter-tight' })

export const metadata: Metadata = {
  title: 'Risk Screener — AI health risk check',
  description:
    'Screen for hypertension, type 2 diabetes, chronic kidney disease, anemia and diabetic retinopathy with a short questionnaire and two photos.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${interTight.variable}`}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
