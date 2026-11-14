import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Same variable the backend reads, so the dev proxy follows a changed port.
const backendPort = process.env.BALLTRACE_PORT ?? '8000'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    restoreMocks: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
})
