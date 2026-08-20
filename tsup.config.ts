import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  platform: 'browser',
  target: 'es2018',
  minify: false,
  treeshake: true,
  outDir: 'dist',
  // Published package depends on npm onnxruntime-web (never CDN).
  external: ['onnxruntime-web', /^onnxruntime-web\//],
  outExtension({ format }) {
    return { js: format === 'cjs' ? '.cjs' : '.js' };
  },
});
