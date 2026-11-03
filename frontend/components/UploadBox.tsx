'use client'

import { useEffect, useId, useState } from 'react'

const MAX_MB = 8
const TYPES = ['image/jpeg', 'image/png', 'image/webp']

interface UploadBoxProps {
  title: string
  checks: React.ReactNode
  tips: React.ReactNode[]
  file: File | null
  onChange: (f: File | null) => void
  sampleUrl?: string
  roundPreview?: boolean
}

export function UploadBox({ title, checks, tips, file, onChange, sampleUrl, roundPreview }: UploadBoxProps) {
  const id = useId()
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    if (!file) return setPreview(null)
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function accept(f: File | undefined) {
    if (!f) return
    if (!TYPES.includes(f.type)) return setError('Please choose a JPEG, PNG or WebP image.')
    if (f.size > MAX_MB * 1024 * 1024) return setError(`Please choose an image under ${MAX_MB} MB.`)
    setError(null)
    onChange(f)
  }

  async function loadSample() {
    if (!sampleUrl) return
    const blob = await (await fetch(sampleUrl)).blob()
    accept(new File([blob], sampleUrl.split('/').pop() ?? 'sample.jpg', { type: blob.type || 'image/jpeg' }))
  }

  return (
    <section className="flex min-w-0 flex-1 basis-[420px] flex-col gap-4 rounded-[20px] border border-line bg-surface p-7">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1.5">
          <h2 className="text-[22px] font-semibold">{title}</h2>
          <span className="text-sm text-ink-2">{checks}</span>
        </div>
        {file ? (
          <span className="flex items-center gap-1.5 rounded-full bg-low-soft px-2.5 py-1 text-[13px] font-semibold text-low">
            <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12l5 5 9-10" /></svg>
            Ready
          </span>
        ) : (
          <span className="rounded-full bg-line-soft px-2.5 py-1 text-[13px] font-semibold text-ink-2">Optional</span>
        )}
      </div>

      {file && preview ? (
        <div className="flex items-center gap-4 rounded-[14px] border border-line p-3.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt={`Preview of ${file.name}`}
            className={`h-24 w-24 flex-none object-cover ${roundPreview ? 'rounded-full' : 'rounded-[10px]'}`} />
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="truncate text-[15px] font-semibold">{file.name}</span>
            <span className="text-[13px] text-muted">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
            <div className="flex gap-4">
              <label htmlFor={id} className="cursor-pointer text-sm font-semibold text-brand">Replace</label>
              <button type="button" onClick={() => onChange(null)} className="text-sm font-semibold text-ink-2">
                Remove
              </button>
            </div>
          </div>
        </div>
      ) : (
        <label
          htmlFor={id}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]) }}
          className={`flex min-h-[124px] cursor-pointer flex-col items-center justify-center gap-2 rounded-[14px] border-2 border-dashed p-5 text-center ${
            dragging ? 'border-brand bg-brand-soft' : 'border-[#9fb3ad] bg-[#f9fbfa]'
          }`}
        >
          <svg aria-hidden="true" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0b6b61"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 16V4" /><path d="M7 9l5-5 5 5" /><path d="M4 20h16" />
          </svg>
          <span className="text-base font-semibold">Drag a photo here or click to upload</span>
          <span className="text-[13px] text-muted">JPEG, PNG or WebP · up to {MAX_MB} MB</span>
        </label>
      )}
      <input id={id} type="file" accept={TYPES.join(',')} className="sr-only"
        onChange={(e) => { accept(e.target.files?.[0]); e.target.value = '' }} />

      {error && <p role="alert" className="text-sm font-medium text-high">{error}</p>}

      <ul className="flex flex-col gap-2 text-sm leading-relaxed text-ink-2">
        {tips.map((t, i) => <li key={i}>{t}</li>)}
        {sampleUrl && !file && (
          <li>
            <button type="button" onClick={loadSample} className="font-semibold text-brand underline">
              Use a sample photo
            </button>
          </li>
        )}
      </ul>
    </section>
  )
}
