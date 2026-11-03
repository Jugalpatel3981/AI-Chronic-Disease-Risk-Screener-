// Turns API responses into the simple rows shown on the results screen, and
// keeps the latest results in sessionStorage between the two pages.

import type { AnemiaResult, QuestionnaireResult, RetinaResult, RiskLevel } from './api'

export type RowLevel = RiskLevel | 'diagnosed' | 'not-checked'

export interface ResultRow {
  key: string
  name: string
  value: string
  level: RowLevel
  note: string
}

export interface StoredResults {
  questionnaire: QuestionnaireResult
  anemia: AnemiaResult | null
  retina: RetinaResult | null
  anemiaError: string | null
  retinaError: string | null
  createdAt: string
}

const KEY = 'risk-screener-results'

export function saveResults(r: StoredResults) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(r))
  } catch {
    /* storage unavailable: the results page will show "no results" */
  }
}

export function loadResults(): StoredResults | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as StoredResults) : null
  } catch {
    return null
  }
}

export function clearResults() {
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    /* ignore */
  }
}

const pct = (p: number) => `${Math.round(p * 100)}%`

export function toRows(r: StoredResults): ResultRow[] {
  const rows: ResultRow[] = r.questionnaire.results.map((d) => ({
    key: d.disease,
    name: d.title,
    value: pct(d.probability),
    level: d.already_diagnosed ? 'diagnosed' : d.risk_level,
    note: d.already_diagnosed
      ? 'You told us you have been diagnosed'
      : d.screen_positive
        ? 'Above the screening threshold'
        : 'Below the screening threshold',
  }))

  rows.push(
    r.anemia
      ? {
          key: 'anemia',
          name: 'Anemia',
          value: pct(r.anemia.probability),
          level: r.anemia.risk_level,
          note: `Estimated haemoglobin ${r.anemia.hemoglobin_g_dl.toFixed(1)} g/dL`,
        }
      : { key: 'anemia', name: 'Anemia', value: '—', level: 'not-checked',
          note: r.anemiaError ?? 'Not checked — no nail photo' },
  )

  rows.push(
    r.retina
      ? {
          key: 'retina',
          name: 'Diabetic Retinopathy',
          value: `Grade ${r.retina.grade}`,
          level: r.retina.risk_level,
          note: r.retina.grade === 0 ? 'No signs found in your retina photo' : r.retina.grade_name,
        }
      : { key: 'retina', name: 'Diabetic Retinopathy', value: '—', level: 'not-checked',
          note: r.retinaError ?? 'Not checked — no retina photo' },
  )
  return rows
}

/** One plain next step, chosen from the most important finding. */
export function nextStep(r: StoredResults): string {
  const flagged = (k: string) => {
    if (k === 'anemia') return r.anemia && r.anemia.risk_level !== 'low'
    if (k === 'retina') return r.retina?.referable
    const d = r.questionnaire.results.find((x) => x.disease === k)
    return d && !d.already_diagnosed && d.risk_level !== 'low'
  }
  if (flagged('retina')) return 'See an eye specialist about your retina result.'
  if (flagged('ckd')) return 'Ask your doctor for a kidney function blood and urine test.'
  if (flagged('diabetes')) return 'Ask your doctor for an HbA1c blood test to check your blood sugar.'
  if (flagged('hypertension')) return 'Get your blood pressure checked — most pharmacies do it for free.'
  if (flagged('anemia')) return 'Ask your doctor for a full blood count to check your haemoglobin.'
  return 'Keep up your healthy habits and repeat this check in about a year.'
}
