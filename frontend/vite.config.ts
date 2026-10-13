import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Fail instead of drifting to 5174: the backend's CORS allowlist pins
    // this exact origin, so a fallback port silently breaks every request.
    strictPort: true,
  },
})
