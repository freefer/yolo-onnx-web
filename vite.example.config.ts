import { cpSync, existsSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

const ORT_CDN_VERSION = '1.30.0';
const ORT_CDN_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist`;

/**
 * Online demo / GitHub Pages only.
 * Library source still imports `onnxruntime-web/*` (npm).
 * This build rewrites those imports to CDN URLs so the published package stays npm-only.
 */
export default defineConfig({
  define: {
    'import.meta.env.YOLO_ORT_SOURCE': JSON.stringify('cdn'),
    'import.meta.env.YOLO_DEMO_MODE': JSON.stringify('pages'),
    'import.meta.env.YOLO_ORT_WASM_PATHS': JSON.stringify(`${ORT_CDN_BASE}/`),
  },
  resolve: {
    alias: {
      'onnxruntime-web/webgpu': `${ORT_CDN_BASE}/ort.webgpu.bundle.min.mjs`,
      'onnxruntime-web/jspi': `${ORT_CDN_BASE}/ort.jspi.bundle.min.mjs`,
      'onnxruntime-web/wasm': `${ORT_CDN_BASE}/ort.wasm.bundle.min.mjs`,
      'onnxruntime-web/webgl': `${ORT_CDN_BASE}/ort.webgl.min.mjs`,
      'onnxruntime-web/all': `${ORT_CDN_BASE}/ort.all.bundle.min.mjs`,
    },
  },
  plugins: [
    {
      name: 'copy-pages-static-assets',
      closeBundle() {
        const modelSource = resolve('examples/model');
        const modelTarget = resolve('dist-example/examples/model');

        if (existsSync(modelSource)) {
          cpSync(modelSource, modelTarget, { recursive: true });
        }

        const coiSource = resolve('node_modules/coi-serviceworker/coi-serviceworker.min.js');
        const coiFallback = resolve('examples/browser/coi-serviceworker.min.js');
        const coiFile = existsSync(coiSource) ? coiSource : coiFallback;
        const coiTargetDir = resolve('dist-example/examples/browser');

        if (existsSync(coiFile)) {
          cpSync(coiFile, resolve(coiTargetDir, 'coi-serviceworker.min.js'));
        }

        writeFileSync(resolve('dist-example/.nojekyll'), '');
      },
    },
  ],
  base: './',
  build: {
    outDir: 'dist-example',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: 'index.html',
        browser: 'examples/browser/index.html',
        sam3: 'examples/browser/sam3.html',
      },
      external: [/^https?:\/\//],
    },
  },
});
