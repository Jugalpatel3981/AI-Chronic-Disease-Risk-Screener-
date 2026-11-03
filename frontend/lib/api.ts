// Typed client for the FastAPI backend. Types mirror backend/app/schemas.py.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000'

export type Sex = 'male' | 'female'
export type RiskLevel = 'low' | 'moderate' | 'high'

export interface Questionnaire {
  age: number
  sex: Sex
  height_cm: number
  weight_kg: number
  waist_cm?: number | null
  exercise_min_week: number
  sitting_hours_day: number
  sleep_hours: number
  alcohol_drinks_week: number
  smoking: 'never' | 'former' | 'current'
  calories_kcal?: number | null
  diagnosed_high_bp: boolean
  diagnosed_diabetes: boolean
}

export interface DiseaseRisk {
  disease: 'hypertension' | 'diabetes' | 'ckd'
  title: string
  probability: number
  risk_level: RiskLevel
  screen_positive: boolean
  threshold: number
  already_diagnosed: boolean
}

export interface QuestionnaireResult {
  bmi: number
  results: DiseaseRisk[]
}

export interface AnemiaResult {
  hemoglobin_g_dl: number
  hemoglobin_range_g_dl: [number, number]
  probability: number
  risk_level: RiskLevel
  threshold_g_dl: number
}

export interface RetinaResult {
  grade: number
  grade_name: string
  probability_referable: number
  risk_level: RiskLevel
  referable: boolean
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError('Could not reach the screening server. Please try again in a moment.', 0)
  }
  if (!res.ok) {
    let message = `Something went wrong (error ${res.status}).`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(message, res.status)
  }
  return res.json() as Promise<T>
}

export function predictQuestionnaire(q: Questionnaire) {
  return request<QuestionnaireResult>('/api/v1/predict/questionnaire', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(q),
  })
}

export function predictAnemia(image: File, sex: Sex) {
  const form = new FormData()
  form.append('image', image)
  form.append('sex', sex)
  return request<AnemiaResult>('/api/v1/predict/anemia', { method: 'POST', body: form })
}

export function predictRetina(image: File) {
  const form = new FormData()
  form.append('image', image)
  return request<RetinaResult>('/api/v1/predict/retina', { method: 'POST', body: form })
}
