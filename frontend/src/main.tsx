import { Component, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

interface RootErrorBoundaryState {
  hasError: boolean
  message: string
}

class RootErrorBoundary extends Component<{ children: ReactNode }, RootErrorBoundaryState> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error: Error): RootErrorBoundaryState {
    return {
      hasError: true,
      message: error?.message ?? 'Unknown runtime error',
    }
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error('[RootErrorBoundary] Frontend crash', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ minHeight: '100svh', background: '#080810', color: '#ff5252', padding: 20, fontFamily: 'JetBrains Mono, monospace' }}>
          <h2 style={{ color: '#ff5252', marginTop: 0 }}>Frontend runtime error</h2>
          <p style={{ color: 'rgba(255,255,255,0.85)' }}>Se capturo un error para evitar pantalla negra.</p>
          <pre style={{ whiteSpace: 'pre-wrap', background: '#130b12', border: '1px solid rgba(255,82,82,0.35)', borderRadius: 8, padding: 12 }}>
            {this.state.message}
          </pre>
        </div>
      )
    }

    return this.props.children
  }
}

createRoot(document.getElementById('root')!).render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>,
)
