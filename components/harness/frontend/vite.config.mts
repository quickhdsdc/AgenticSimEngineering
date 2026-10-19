import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'
import { existsSync } from 'node:fs'

// The independent workspace keeps build output in work/. A standalone
// component/Docker build retains dist, as required by its existing Dockerfile.
const projectCheckout = existsSync(fileURLToPath(new URL('../../../pyproject.toml', import.meta.url)))

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: projectCheckout ? fileURLToPath(new URL('../../../work/build/harness-frontend', import.meta.url)) : 'dist'
  },
  server: {
    port: 5173
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
    extensions: ['.js', '.ts', '.vue', '.json']
  }
})
