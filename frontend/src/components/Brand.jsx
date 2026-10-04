import { AudioLines } from 'lucide-react'

export function Brand({ compact = false }) {
  return <a className={`brand ${compact ? 'brand-compact' : ''}`} href="#home" aria-label="DubFlow home">
    <span className="brand-mark"><AudioLines size={19} strokeWidth={2.2} /></span>
    <span>dubflow<span className="brand-period">.</span></span>
  </a>
}

export function Header({ connected }) {
  return <header className="topbar">
    <Brand />
    <div className="topbar-right">
      <span className="workflow-note"><span className={`status-dot ${connected ? 'is-online' : ''}`} />{connected ? 'Studio is ready' : 'Connecting to studio'}</span>
      <span className="topbar-divider" />
      <span className="beta-tag">BETA</span>
    </div>
  </header>
}
