import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vite';

/**
 * Local development only.
 * Uses npm `onnxruntime-web` + local `examples/browser/ort-wasm/`.
 */
export default defineConfig({
  plugins: [basicSsl()],
  define: {
    'import.meta.env.YOLO_ORT_SOURCE': JSON.stringify('npm'),
    'import.meta.env.YOLO_DEMO_MODE': JSON.stringify('local'),
    'import.meta.env.YOLO_ORT_WASM_PATHS': JSON.stringify(''),
  },
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
