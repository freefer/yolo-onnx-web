import basicSsl from '@vitejs/plugin-basic-ssl';
import { cpSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    basicSsl(),
    {
      name: 'copy-example-static-assets',
      closeBundle() {
        const assets = [
          ['examples/browser/ort-wasm', 'dist-example/examples/browser/ort-wasm'],
          ['examples/model', 'dist-example/examples/model'],
        ] as const;

        for (const [from, to] of assets) {
          const source = resolve(from);

          if (existsSync(source)) {
            cpSync(source, resolve(to), { recursive: true });
          }
        }

        const generatedAssetsDir = resolve('dist-example/assets');

        if (existsSync(generatedAssetsDir)) {
          for (const file of readdirSync(generatedAssetsDir)) {
            if (/^ort-wasm.*\.wasm$/.test(file)) {
              rmSync(resolve(generatedAssetsDir, file));
            }
          }
        }
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
      },
    },
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
