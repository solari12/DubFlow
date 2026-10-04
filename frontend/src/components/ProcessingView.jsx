import { Check, LoaderCircle, Sparkles } from 'lucide-react'

const STAGES = [
  ['UPLOAD', 'Video uploaded'], ['ASR', 'Listening to every word'],
  ['DIARIZATION', 'Finding the speakers'], ['TRANSLATION', 'Making it sound natural'],
  ['TTS', 'Giving it a new voice'], ['ALIGNMENT', 'Matching the timing'], ['RENDERING', 'Putting it all together'],
]

export default function ProcessingView({ file, job }) {
  const active = STAGES.findIndex(([key]) => key === job?.stage)
  return <section className="work-card processing-card">
    <div className="processing-art"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="processing-core"><Sparkles size={27} /></div><div className="soundbars">{[13, 27, 18, 35, 24, 42, 20, 32, 15].map((height, index) => <i key={index} style={{ '--bar-h': `${height}px`, '--bar-i': index }} />)}</div></div>
    <span className="eyebrow"><span className="live-pulse" /> THE STUDIO IS WORKING</span>
    <h2>Your new voice<br />is taking shape.</h2>
    <p className="processing-file">{file?.name}</p>
    <div className="progress-copy"><span>{job?.stage === 'RENDERING' ? 'Finishing your video' : `Step ${Math.max(active, 1)} of ${STAGES.length - 1}`}</span><span>{job?.progress || 0}%</span></div>
    <div className="progress-track"><span style={{ width: `${job?.progress || 0}%` }} /></div>
    <div className="stage-list">{STAGES.slice(1).map(([key, label], index) => {
      const stageIndex = index + 1
      const done = active > stageIndex
      const current = active === stageIndex
      return <div key={key} className={`stage-item ${done ? 'stage-done' : ''} ${current ? 'stage-current' : ''}`}>
        <span className="stage-icon">{done ? <Check size={14} /> : current ? <LoaderCircle className="spin" size={14} /> : <span />}</span>
        <span>{label}</span>{current && <small>IN PROGRESS</small>}
      </div>
    })}</div>
    {job?.error && <p className="inline-error" role="alert">{job.error}</p>}
  </section>
}
