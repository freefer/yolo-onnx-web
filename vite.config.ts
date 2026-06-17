import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [basicSsl()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    open: '/examples/browser/',
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
});
