import type * as OrtTypes from 'onnxruntime-web';
import type { OnnxRuntimeWebOptions, YoloExecutionProvider, YoloOptions } from './types';

export type OrtBundle = 'auto' | 'webgpu' | 'wasm' | 'webgl' | 'all';
export type OrtModule = typeof import('onnxruntime-web/webgpu');

let ortModule: OrtModule | null = null;
let loadedBundle: Exclude<OrtBundle, 'auto'> | null = null;
let loadingPromise: Promise<OrtModule> | null = null;

/**
 * Resolve which onnxruntime-web entry to load.
 *
 * - webgpu → `onnxruntime-web/webgpu` (native WebGPU EP, also supports wasm)
 * - webnn / webgl → `onnxruntime-web/all` (JSEP; required for WebNN)
 * - otherwise → `onnxruntime-web/wasm`
 *
 * Do not use `/all` for WebGPU: its JSEP path is legacy and fails on some models
 * (e.g. RT-DETR MaxPool ceil_mode).
 */
export function resolveOrtBundle(
  executionProviders: readonly YoloExecutionProvider[] = ['wasm'],
  ortBundle: OrtBundle = 'auto',
): Exclude<OrtBundle, 'auto'> {
  if (ortBundle !== 'auto') {
    return ortBundle;
  }

  const names = new Set(
    executionProviders.map(provider => {
      if (typeof provider === 'string') {
        return provider;
      }

      return (provider as { name?: string }).name ?? '';
    }),
  );

  if (names.has('webgpu')) {
    return 'webgpu';
  }
  if (names.has('webgl')) {
    return 'webgl';
  }

  if (names.has('webnn')) {
    return 'all';
  }

  return 'wasm';
}

/** Whether an already-loaded ORT entry can serve the requested entry without a page reload. */
export function canReuseOrtBundle(
  loaded: Exclude<OrtBundle, 'auto'>,
  requested: Exclude<OrtBundle, 'auto'>,
): boolean {
  if (loaded === requested) {
    return true;
  }

  // Native WebGPU bundle also includes wasm EP.
  if (loaded === 'webgpu' && requested === 'wasm') {
    return true;
  }

  // `/all` covers wasm/webgl/webnn (and JSEP webgpu).
  if (loaded === 'all' && (requested === 'wasm' || requested === 'webgl')) {
    return true;
  }

  return false;
}

async function importOrtBundle(bundle: Exclude<OrtBundle, 'auto'>): Promise<OrtModule> {
  switch (bundle) {
    case 'webgpu':
      return import('onnxruntime-web/webgpu');
    case 'webgl':
      return import('onnxruntime-web/webgl');
    case 'all':
      return import('onnxruntime-web/all');
    case 'wasm':
    default:
      return import('onnxruntime-web/wasm');
  }
}

function applyOnnxRuntimeWebOptions(ort: OrtModule, options: OnnxRuntimeWebOptions): void {
  if (options.wasmPaths !== undefined) {
    ort.env.wasm.wasmPaths = options.wasmPaths as typeof ort.env.wasm.wasmPaths;
  }

  if (options.numThreads !== undefined) {
    ort.env.wasm.numThreads = options.numThreads;
  }

  if (options.proxy !== undefined) {
    ort.env.wasm.proxy = options.proxy;
  }
}

function resolveRequestedBundle(options: YoloOptions): Exclude<OrtBundle, 'auto'> {
  const executionProviders =
    options.sessionOptions?.executionProviders ??
    options.executionProviders ??
    (['wasm'] as const);

  return resolveOrtBundle(executionProviders, options.ortBundle ?? 'auto');
}

/**
 * Configure onnxruntime-web before creating an inference session.
 * Lazily loads the matching package entry from `executionProviders` / `ortBundle`.
 */
export async function initializeOnnxRuntimeWeb(options: YoloOptions = {}): Promise<OrtModule> {
  const bundle = resolveRequestedBundle(options);

  if (ortModule && loadedBundle) {
    if (canReuseOrtBundle(loadedBundle, bundle)) {
      applyOnnxRuntimeWebOptions(ortModule, options);
      return ortModule;
    }

    const error = new Error(
      `onnxruntime-web is already loaded as "${loadedBundle}", but "${bundle}" was requested. ` +
        'Reload the page before switching between WebGPU (native) and WebNN/WebGL (all/JSEP) bundles.',
    ) as Error & { code: string; loadedBundle: string; requestedBundle: string };
    error.code = 'ORT_BUNDLE_RELOAD_REQUIRED';
    error.loadedBundle = loadedBundle;
    error.requestedBundle = bundle;
    throw error;
  }

  if (!loadingPromise) {
    loadingPromise = importOrtBundle(bundle).then(module => {
      ortModule = module;
      loadedBundle = bundle;
      loadingPromise = null;
      return module;
    });
  }

  const module = await loadingPromise;
  applyOnnxRuntimeWebOptions(module, options);
  return module;
}

export async function ensureOnnxRuntimeWebInitialized(options: YoloOptions = {}): Promise<OrtModule> {
  return initializeOnnxRuntimeWeb(options);
}

export function getOrt(): OrtModule {
  if (!ortModule) {
    throw new Error('ONNX Runtime Web is not initialized. Call Yolo.create() / load() first.');
  }

  return ortModule;
}

export function getLoadedOrtBundle(): Exclude<OrtBundle, 'auto'> | null {
  return loadedBundle;
}

/** Compatible alias used after initialization (Tensor / InferenceSession / env). */
export const ort: OrtModule = new Proxy({} as OrtModule, {
  get(_target, property, receiver) {
    return Reflect.get(getOrt(), property, receiver);
  },
}) as OrtModule;

export type { OrtTypes };
