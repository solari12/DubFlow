import { ArrowDownToLine, ArrowLeft, Check, Clapperboard, Share2, Sparkles } from 'lucide-react'
import { getArtifactUrl } from '../api/client.js'

export default function ResultView({ jobId, onReset }) {
  const videoUrl = getArtifactUrl(jobId)
  async function share() {
    const artifactUrl = window.location.origin + videoUrl
    if (navigator.share) await navigator.share({ title: 'My dubbed video', url: artifactUrl })
    else if (navigator.clipboard) await navigator.clipboard.writeText(artifactUrl)
  }
  return <section className="result-layout">
    <div className="result-main"><video className="video-preview" controls src={videoUrl} />
      <div className="video-caption"><div className="video-caption-icon"><Clapperboard size={18} /></div><div><strong>Your video, in a new voice.</strong><span>DubFlow Studio · Just now</span></div><span className="complete-pill"><Check size={13} /> READY</span></div>
    </div>
    <aside className="result-side"><span className="eyebrow"><Sparkles size={13} /> THAT'S A WRAP</span><h2>Made to be<br />heard everywhere.</h2><p>Your dubbed video is ready. Give it a watch, then send it out into the world.</p>
      <a className="primary-button download-button" href={videoUrl} download="dubflow-dubbed.mp4"><ArrowDownToLine size={18} /> Download video</a>
      <button type="button" className="secondary-button" onClick={share}><Share2 size={17} /> Share your video</button>
      <button type="button" className="text-button" onClick={onReset}><ArrowLeft size={15} /> Dub another video</button>
    </aside>
  </section>
}
