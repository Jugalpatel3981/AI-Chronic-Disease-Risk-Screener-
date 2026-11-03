'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { ChoiceGroup, NumberField } from '@/components/Fields'
import { Header, buttonPrimary, buttonSecondary } from '@/components/Header'
import { UploadBox } from '@/components/UploadBox'
import { ApiError, predictAnemia, predictQuestionnaire, predictRetina, type Questionnaire, type Sex } from '@/lib/api'
import { saveResults } from '@/lib/results'

const STEPS = ['About you', 'Lifestyle', 'Health history', 'Photos'] as const

type Form = {
  age: string; sex: Sex | ''; height: string; weight: string
  exercise: string; sitting: string; sleep: string; alcohol: string
  smoking: Questionnaire['smoking'] | ''; calories: string
  waist: string; highBp: 'yes' | 'no' | ''; diabetes: 'yes' | 'no' | ''
}

const EMPTY: Form = {
  age: '', sex: '', height: '', weight: '', exercise: '', sitting: '', sleep: '', alcohol: '',
  smoking: '', calories: '', waist: '', highBp: '', diabetes: '',
}

type Errors = Partial<Record<keyof Form, string>>

function range(v: string, lo: number, hi: number, label: string, optional = false): string | undefined {
  if (v.trim() === '') return optional ? undefined : `Please enter your ${label}.`
  const n = Number(v)
  if (!Number.isFinite(n) || n < lo || n > hi) return `Please enter a value between ${lo} and ${hi}.`
}

function validate(step: number, f: Form): Errors {
  const e: Errors = {}
  if (step === 0) {
    e.age = range(f.age, 18, 120, 'age')
    if (!f.sex) e.sex = 'Please choose one.'
    e.height = range(f.height, 100, 230, 'height')
    e.weight = range(f.weight, 25, 300, 'weight')
    if (!e.height && !e.weight) {
      const bmi = Number(f.weight) / (Number(f.height) / 100) ** 2
      if (bmi < 10 || bmi > 90) e.weight = 'Please check your height and weight.'
    }
  }
  if (step === 1) {
    e.exercise = range(f.exercise, 0, 10080, 'exercise time')
    e.sitting = range(f.sitting, 0, 24, 'sitting time')
    e.sleep = range(f.sleep, 0, 24, 'sleep time')
    e.alcohol = range(f.alcohol, 0, 200, 'number of drinks')
    if (!f.smoking) e.smoking = 'Please choose one.'
    e.calories = range(f.calories, 300, 10000, 'calories', true)
  }
  if (step === 2) {
    e.waist = range(f.waist, 40, 200, 'waist size', true)
    if (!f.highBp) e.highBp = 'Please choose one.'
    if (!f.diabetes) e.diabetes = 'Please choose one.'
  }
  return Object.fromEntries(Object.entries(e).filter(([, v]) => v)) as Errors
}

function toQuestionnaire(f: Form): Questionnaire {
  return {
    age: Number(f.age), sex: f.sex as Sex, height_cm: Number(f.height), weight_kg: Number(f.weight),
    waist_cm: f.waist ? Number(f.waist) : null,
    exercise_min_week: Number(f.exercise), sitting_hours_day: Number(f.sitting), sleep_hours: Number(f.sleep),
    alcohol_drinks_week: Number(f.alcohol), smoking: f.smoking as Questionnaire['smoking'],
    calories_kcal: f.calories ? Number(f.calories) : null,
    diagnosed_high_bp: f.highBp === 'yes', diagnosed_diabetes: f.diabetes === 'yes',
  }
}

export default function Screen() {
  const router = useRouter()
  const [step, setStep] = useState(0)
  const [form, setForm] = useState<Form>(EMPTY)
  const [errors, setErrors] = useState<Errors>({})
  const [nail, setNail] = useState<File | null>(null)
  const [retina, setRetina] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [failure, setFailure] = useState<string | null>(null)

  const set = <K extends keyof Form>(k: K) => (v: Form[K]) => setForm((f) => ({ ...f, [k]: v }))

  function next() {
    const e = validate(step, form)
    setErrors(e)
    if (Object.keys(e).length === 0) {
      setStep((s) => s + 1)
      window.scrollTo({ top: 0 })
    }
  }

  async function analyze() {
    setBusy(true)
    setFailure(null)
    const message = (r: PromiseSettledResult<unknown>) =>
      r.status === 'rejected' ? (r.reason instanceof ApiError ? r.reason.message : 'This photo could not be analysed.') : null
    try {
      const [q, a, r] = await Promise.allSettled([
        predictQuestionnaire(toQuestionnaire(form)),
        nail ? predictAnemia(nail, form.sex as Sex) : Promise.resolve(null),
        retina ? predictRetina(retina) : Promise.resolve(null),
      ])
      if (q.status === 'rejected') throw q.reason
      saveResults({
        questionnaire: q.value,
        anemia: a.status === 'fulfilled' ? a.value : null,
        retina: r.status === 'fulfilled' ? r.value : null,
        anemiaError: message(a),
        retinaError: message(r),
        createdAt: new Date().toISOString(),
      })
      router.push('/results')
    } catch (err) {
      setFailure(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.')
      setBusy(false)
    }
  }

  const progress = `${Math.round(((step + 1) / (STEPS.length + 0.5)) * 100)}%`

  return (
    <div className="min-h-screen">
      <Header right={`Step ${step + 1} of ${STEPS.length} · ${STEPS[step]}`} />
      <main className="mx-auto flex max-w-[1160px] flex-col gap-8 px-6 pt-10 pb-16">
        <nav aria-label="Screening progress" className="flex flex-col gap-3">
          <div className="h-1.5 overflow-hidden rounded-full bg-line">
            <div className="h-1.5 rounded-full bg-brand transition-all" style={{ width: progress }} />
          </div>
          <ol className="flex flex-wrap gap-x-5 gap-y-2 text-sm">
            {STEPS.map((s, i) => (
              <li key={s} aria-current={i === step ? 'step' : undefined}
                className={`flex items-center gap-2 ${i === step ? 'font-semibold text-ink' : i < step ? 'text-ink-2' : 'text-muted'}`}>
                <span className={`flex h-[22px] w-[22px] items-center justify-center rounded-full text-xs font-semibold ${
                  i < step ? 'bg-brand text-white' : i === step ? 'bg-ink text-white' : 'bg-line text-ink-2'}`}>
                  {i < step ? '✓' : i + 1}
                </span>
                {s}
              </li>
            ))}
          </ol>
        </nav>

        {step < 3 ? (
          <form onSubmit={(e) => { e.preventDefault(); next() }} noValidate
            className="flex flex-col gap-7 rounded-[20px] border border-line bg-surface p-6 sm:p-8">
            {step === 0 && (
              <>
                <h1 className="font-display text-3xl font-bold">About you</h1>
                <div className="flex flex-wrap gap-6">
                  <NumberField label="Age" unit="years" value={form.age} onChange={set('age')} error={errors.age} step="1" />
                  <NumberField label="Height" unit="cm" value={form.height} onChange={set('height')} error={errors.height} />
                  <NumberField label="Weight" unit="kg" value={form.weight} onChange={set('weight')} error={errors.weight} />
                </div>
                <ChoiceGroup legend="Sex" value={form.sex} onChange={set('sex')} error={errors.sex}
                  options={[{ value: 'female', label: 'Female' }, { value: 'male', label: 'Male' }]} />
              </>
            )}
            {step === 1 && (
              <>
                <div className="flex flex-col gap-2">
                  <h1 className="font-display text-3xl font-bold">Your lifestyle</h1>
                  <p className="text-[15px] text-ink-2">Think about a typical week. Rough estimates are fine.</p>
                </div>
                <div className="flex flex-wrap gap-6">
                  <NumberField label="Exercise per week" unit="minutes" value={form.exercise} onChange={set('exercise')}
                    error={errors.exercise} help="Activity that makes you breathe harder: brisk walking, cycling, sport." />
                  <NumberField label="Sitting per day" unit="hours" value={form.sitting} onChange={set('sitting')}
                    error={errors.sitting} help="At work, commuting and watching screens." />
                  <NumberField label="Sleep per night" unit="hours" value={form.sleep} onChange={set('sleep')} error={errors.sleep} />
                  <NumberField label="Alcoholic drinks per week" unit="drinks" value={form.alcohol} onChange={set('alcohol')}
                    error={errors.alcohol} help="One drink = a small glass of wine, a bottle of beer or one shot." />
                </div>
                <ChoiceGroup legend="Do you smoke?" value={form.smoking} onChange={set('smoking')} error={errors.smoking}
                  options={[{ value: 'never', label: 'Never' }, { value: 'former', label: 'I used to' }, { value: 'current', label: 'Yes, currently' }]} />
                <div className="max-w-[360px]">
                  <NumberField label="Daily calories" unit="kcal" optional value={form.calories} onChange={set('calories')}
                    error={errors.calories} placeholder="Skip if you're not sure" />
                </div>
              </>
            )}
            {step === 2 && (
              <>
                <h1 className="font-display text-3xl font-bold">Health history</h1>
                <ChoiceGroup legend="Has a doctor ever told you that you have high blood pressure?" value={form.highBp}
                  onChange={set('highBp')} error={errors.highBp}
                  options={[{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }]} />
                <ChoiceGroup legend="Has a doctor ever told you that you have diabetes?" value={form.diabetes}
                  onChange={set('diabetes')} error={errors.diabetes}
                  options={[{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }]} />
                <div className="max-w-[360px]">
                  <NumberField label="Waist size" unit="cm" optional value={form.waist} onChange={set('waist')}
                    error={errors.waist} help="Measure around your belly button." />
                </div>
              </>
            )}
            <div className="flex flex-wrap justify-between gap-3 border-t border-line-soft pt-6">
              {step === 0
                ? <Link href="/" className={buttonSecondary}>Back</Link>
                : <button type="button" onClick={() => setStep((s) => s - 1)} className={buttonSecondary}>Back</button>}
              <button type="submit" className={buttonPrimary}>Continue</button>
            </div>
          </form>
        ) : (
          <>
            <div className="flex flex-col gap-2">
              <h1 className="font-display text-3xl font-bold">Add your photos</h1>
              <p className="text-base text-ink-2">Both are optional. Each photo adds one more check to your results.</p>
            </div>
            <div className="flex flex-wrap gap-6">
              <UploadBox title="Nail photo" checks={<>Checks for <strong>anemia</strong></>} file={nail} onChange={setNail}
                sampleUrl="/samples/nail.jpg"
                tips={['Fingers flat on white paper, nails clearly visible', 'Daylight, no nail polish']} />
              <UploadBox title="Retina photo" checks={<>Checks for <strong>diabetic retinopathy</strong></>} file={retina}
                onChange={setRetina} sampleUrl="/samples/retina.jpg" roundPreview
                tips={['A fundus photo from an eye clinic or optician — not a phone photo of your eye']} />
            </div>
            {failure && (
              <p role="alert" className="rounded-xl border border-high/30 bg-high-soft px-4 py-3 text-[15px] font-medium text-high">
                {failure}
              </p>
            )}
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-6">
              <button type="button" onClick={() => setStep(2)} className={buttonSecondary} disabled={busy}>Back</button>
              <div className="flex flex-wrap items-center gap-4">
                <span className="text-sm text-muted">Photos are analysed in memory and never stored</span>
                <button type="button" onClick={analyze} className={`${buttonPrimary} min-h-14 px-8 text-base`} disabled={busy}>
                  {busy ? 'Analyzing…' : 'Analyze my results'}
                </button>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
