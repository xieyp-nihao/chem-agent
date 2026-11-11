import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // 后端没有 /api 前缀；去掉代理前缀后再转发，否则会 404
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
