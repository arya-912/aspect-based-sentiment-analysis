import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
// `defineConfig` from vitest/config rather than vite: same function, plus the
// types for the `test` block below.
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Fail instead of drifting to 5174: the backend's CORS allowlist pins
    // this exact origin, so a fallback port silently breaks every request.
    strictPort: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // `describe`/`it`/`expect` without an import in every file, matching the
    // jest-dom matchers the setup file registers globally.
    globals: true,
    include: ['src/**/*.test.{ts,tsx}'],
    // Tailwind's Vite plugin would compile the whole stylesheet for every
    // test file; no assertion here reads a computed style.
    css: false,
    restoreMocks: true,
  },
})
