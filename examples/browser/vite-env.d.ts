/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** npm for local, cdn for GitHub Pages demo build only */
  readonly YOLO_ORT_SOURCE?: 'npm' | 'cdn';
  readonly YOLO_DEMO_MODE?: 'local' | 'pages';
  /** Empty for local (use ./ort-wasm/). CDN dist URL for Pages demo. */
  readonly YOLO_ORT_WASM_PATHS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
