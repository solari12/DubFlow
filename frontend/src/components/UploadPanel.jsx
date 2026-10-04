import { useRef, useState } from 'react'
import { ArrowRight, Check, FileVideo2, Globe2, Mic2, Sparkles, Upload, X } from 'lucide-react'

const ACCEPT = '.mp4,.mov,.mkv,.webm,video/mp4,video/quicktime,video/x-matroska,video/webm'
const voiceGroups = [
  {
    label: 'Female',
    voices: [
      { value: 'bao_kim', label: 'Bao Kim' },
      { value: 'khanh_vy', label: 'Khanh Vy' },
      { value: 'ngoc_huyen', label: 'Ngoc Huyen' },
      { value: 'phuong_linh', label: 'Phuong Linh' },
      { value: 'quynh_nhu', label: 'Quynh Nhu' },
    ],
  },
  {
    label: 'Male',
    voices: [
      { value: 'gia_bao', label: 'Gia Bao' },
      { value: 'hoang_nam', label: 'Hoang Nam' },
      { value: 'huu_dat', label: 'Huu Dat' },
      { value: 'quang_huy', label: 'Quang Huy' },
      { value: 'thanh_phong', label: 'Thanh Phong' },
    ],
  },
]
const languages = [
  { value: 'auto', label: 'Auto-detect' }, { value: 'en', label: 'English' },
  { value: 'ja', label: 'Japanese' }, { value: 'ko', label: 'Korean' },
  { value: 'zh', label: 'Chinese' }, { value: 'vi', label: 'Vietnamese' },
]

function formatSize(bytes) {
  return bytes > 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

export default function UploadPanel({ onStart, busy }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [sourceLanguage, setSourceLanguage] = useState('auto')
  const [targetLanguage, setTargetLanguage] = useState('vi')
  const [voice, setVoice] = useState('gia_bao')
  const [error, setError] = useState('')

  function choose(candidate) {
    if (!candidate) return
    if (!['.mp4', '.mov', '.mkv', '.webm'].includes(`.${candidate.name.split('.').pop().toLowerCase()}`)) {
      setError('Choose an MP4, MOV, MKV, or WebM video.'); return
    }
    setError(''); setFile(candidate)
  }

  function start() {
    if (!file) return
    onStart(file, { sourceLanguage, targetLanguage, voice }).catch((reason) => setError(reason.message))
  }

  return <section className="upload-panel" aria-labelledby="upload-title">
    <div className="panel-heading">
      <div><span className="eyebrow"><Sparkles size={13} /> YOUR NEXT VIDEO, REIMAGINED</span>
        <h2 id="upload-title">Give your video<br className="mobile-break" /> a new voice.</h2>
        <p>Dub your video into another language while keeping the original visuals intact.</p>
      </div>
      <div className="step-counter"><span>01</span><span className="step-line" /><span>03</span></div>
    </div>

    <div className={`dropzone ${dragging ? 'dropzone-active' : ''} ${file ? 'dropzone-has-file' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files[0]) }}>
      {file ? <div className="file-ready">
        <div className="file-icon"><FileVideo2 size={23} /></div>
        <div className="file-details"><strong>{file.name}</strong><span>{formatSize(file.size)} <i /> Ready to dub</span></div>
        <button type="button" className="icon-button" onClick={() => setFile(null)} aria-label="Remove selected video"><X size={18} /></button>
      </div> : <div className="dropzone-inner">
        <div className="upload-symbol"><Upload size={23} strokeWidth={1.8} /></div>
        <strong>Drop your video here</strong>
        <span className="drop-hint">or <button type="button" onClick={() => inputRef.current?.click()}>browse files</button> from your device</span>
        <span className="file-types">MP4, MOV, MKV or WEBM <i /> UP TO 2 GB</span>
        <input ref={inputRef} hidden type="file" accept={ACCEPT} onChange={(event) => choose(event.target.files?.[0])} />
      </div>}
    </div>

    <div className="settings-row">
      <label className="field"><span><Globe2 size={14} /> FROM</span><select value={sourceLanguage} onChange={(event) => setSourceLanguage(event.target.value)}>{languages.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
      <div className="language-arrow"><ArrowRight size={17} /></div>
      <label className="field"><span><Globe2 size={14} /> INTO</span><select value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)}>{languages.filter((item) => item.value === 'vi').map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
      <label className="field voice-field"><span><Mic2 size={14} /> Voice</span><select aria-label="Voice" value={voice} onChange={(event) => setVoice(event.target.value)}>{voiceGroups.map((group) => <optgroup key={group.label} label={group.label}>{group.voices.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</optgroup>)}</select></label>
    </div>

    {error && <p className="inline-error" role="alert">{error}</p>}
    <button type="button" className="primary-button" disabled={!file || busy} onClick={start}>
      {busy ? 'Preparing your video…' : 'Start dubbing'} {!busy && <ArrowRight size={18} />}
    </button>
    <div className="privacy-note"><Check size={13} /> Your video stays yours. Uploads are processed locally in this prototype.</div>
  </section>
}
