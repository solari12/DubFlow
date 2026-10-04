import { useEffect, useState } from 'react'
import { ArrowUpRight, Instagram, Youtube } from 'lucide-react'
import { createJob, getJob, healthCheck } from './api/client.js'
import { Header } from './components/Brand.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import ProcessingView from './components/ProcessingView.jsx'
import ResultView from './components/ResultView.jsx'

export default function App() {
  const [connected, setConnected] = useState(false)
  const [phase, setPhase] = useState('upload')
  const [file, setFile] = useState(null)
  const [job, setJob] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => { healthCheck().then(() => setConnected(true)).catch(() => setConnected(false)) }, [])
  useEffect(() => {
    if (phase !== 'processing' || !job?.jobId) return undefined
    let stopped = false
    let timer
    const poll = async () => {
      try {
        const next = await getJob(job.jobId)
        if (stopped) return
        setJob(next)
        if (next.status === 'COMPLETED') setPhase('result')
        else if (next.status === 'FAILED') setPhase('failed')
        else timer = window.setTimeout(poll, 1000)
      } catch (error) {
        if (!stopped) { setJob((previous) => ({ ...previous, error: error.message })); timer = window.setTimeout(poll, 1800) }
      }
    }
    timer = window.setTimeout(poll, 500)
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [phase, job?.jobId])

  async function start(video, options) {
    setSubmitting(true); setFile(video)
    try {
      const created = await createJob(video, options)
      setJob(created); setPhase('processing')
    } finally { setSubmitting(false) }
  }

  function reset() { setPhase('upload'); setFile(null); setJob(null) }

  return <div className="app-shell" id="home">
    <Header connected={connected} />
    <main>
      <section className="hero-intro">
        <div className="hero-copy"><div className="creator-tag"><span className="tag-star">✳</span> A LITTLE MORE YOU, EVERYWHERE</div>
          <h1>Good videos<br /><em>travel.</em></h1>
          <p>Give your videos a voice that speaks to everyone.<br className="desktop-break" /> Your story deserves to be understood.</p>
        </div>
        <div className="hero-decoration" aria-hidden="true"><div className="deco-disc"><span>DF</span><i /><b /></div><span className="deco-caption">SOUND LIKE YOU,<br />IN EVERY LANGUAGE</span></div>
      </section>

      {phase === 'upload' && <UploadPanel onStart={start} busy={submitting} />}
      {phase === 'processing' && <ProcessingView file={file} job={job} />}
      {phase === 'failed' && <section className="work-card failed-card"><span className="eyebrow">A SMALL DETOUR</span><h2>We couldn't finish this one.</h2><p>{job?.error || 'The dubbing job failed. Check that the development API has the required AI models installed.'}</p><button className="secondary-button" onClick={reset}>Try another video</button></section>}
      {phase === 'result' && <ResultView jobId={job.jobId} onReset={reset} />}

      <section className="how-row" aria-label="How DubFlow works"><span className="how-label">THREE STEPS TO EVERYWHERE</span><div className="how-item"><span>01</span> Add your video</div><i /><div className="how-item"><span>02</span> Choose a language</div><i /><div className="how-item"><span>03</span> Share your story</div></section>
    </main>
    <footer className="footer"><span>© 2026 DubFlow Studio</span><span className="footer-message">Made for stories worth sharing <span>♥</span></span><div className="social-links"><a href="https://www.youtube.com" aria-label="YouTube"><Youtube size={17} /></a><a href="https://www.instagram.com" aria-label="Instagram"><Instagram size={16} /></a><a href="#home" aria-label="Back to top"><ArrowUpRight size={16} /></a></div></footer>
  </div>
}
