import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    // The Dev API binds to IPv4 127.0.0.1 on Windows; avoiding `localhost`
    // here prevents Node's IPv6-first resolution from producing ECONNREFUSED.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
