import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import './index.css'

const container = document.getElementById('root')

// Checked rather than asserted with `!`: if index.html ever loses the mount
// point, this says so instead of failing with a null dereference inside React.
if (container === null) {
  throw new Error('Mount point #root not found in index.html')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
