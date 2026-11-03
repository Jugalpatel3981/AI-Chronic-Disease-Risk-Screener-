'use client'

import { useId } from 'react'

interface NumberFieldProps {
  label: string
  unit: string
  value: string
  onChange: (v: string) => void
  help?: string
  optional?: boolean
  error?: string
  step?: string
  placeholder?: string
}

export function NumberField({ label, unit, value, onChange, help, optional, error, step, placeholder }: NumberFieldProps) {
  const id = useId()
  return (
    <div className="flex min-w-[240px] flex-1 flex-col gap-2">
      <label htmlFor={id} className="text-[15px] font-semibold">
        {label} {optional && <span className="font-normal text-muted">(optional)</span>}
      </label>
      <div
        className={`flex min-h-12 items-center gap-2 rounded-[10px] border bg-surface px-3.5 focus-within:border-brand ${
          error ? 'border-high' : 'border-line-strong'
        }`}
      >
        <input
          id={id}
          type="number"
          inputMode="decimal"
          step={step ?? 'any'}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error || help ? `${id}-desc` : undefined}
          className="min-w-0 flex-1 bg-transparent text-base outline-none"
        />
        <span className="text-sm text-muted">{unit}</span>
      </div>
      {(error || help) && (
        <span id={`${id}-desc`} className={`text-[13px] ${error ? 'font-medium text-high' : 'text-muted'}`}>
          {error ?? help}
        </span>
      )}
    </div>
  )
}

interface ChoiceProps<T extends string> {
  legend: string
  options: { value: T; label: string }[]
  value: T | ''
  onChange: (v: T) => void
  error?: string
}

export function ChoiceGroup<T extends string>({ legend, options, value, onChange, error }: ChoiceProps<T>) {
  const name = useId()
  return (
    <fieldset className="flex flex-col gap-2.5">
      <legend className="mb-2.5 text-[15px] font-semibold">{legend}</legend>
      <div className="flex flex-wrap gap-2.5">
        {options.map((o) => {
          const selected = value === o.value
          return (
            <label
              key={o.value}
              className={`flex min-h-12 min-w-[150px] flex-1 cursor-pointer items-center gap-2.5 rounded-[10px] px-3.5 text-[15px] ${
                selected ? 'border-2 border-brand bg-[#f0f8f6] font-semibold' : 'border border-line-strong bg-surface'
              }`}
            >
              <input type="radio" name={name} checked={selected} onChange={() => onChange(o.value)}
                className="accent-[#0b6b61]" />
              {o.label}
            </label>
          )
        })}
      </div>
      {error && <span className="text-[13px] font-medium text-high">{error}</span>}
    </fieldset>
  )
}
