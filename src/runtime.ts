import type * as OrtTypes from 'onnxruntime-web';
import type { OnnxRuntimeWebOptions, YoloExecutionProvider, YoloOptions } from './types';

export type OrtBundle = 'auto' | 'webgpu' | 'jspi' | 'wasm' | 'webgl' | 'all';
export type OrtModule = typeof import('onnxruntime-web/webgpu');

let ortModule: OrtModule | null = null;
let loadedBundle: Exclude<OrtBundle, 'auto'> | null = null;
let loadingPromise: Promise<OrtModule> | null = null;

/**
 * Chrome 137+ / Edge expose JSPI as `WebAssembly.Suspending`.
 * The JSPI WebGPU build avoids Asyncify, which otherwise blocks the UI during SAM3 encode.
 */
export function isWebAssemblyJspiAvailable(): boolean {
  const wasm = globalThis.WebAssembly as (typeof WebAssembly & { Suspending?: unknown }) | undefined;
  return Boolean(wasm && 'Suspending' in wasm);
}

/**
 * Resolve which onnxruntime-web entry to load.
 *
 * - webgpu → `onnxruntime-web/jspi` when JSPI is available, otherwise `onnxruntime-web/webgpu` (Asyncify)
 * - webgl → `onnxruntime-web/webgl`
 * - webnn → `onnxruntime-web/all` (JSEP; required for WebNN)
 * - otherwise → `onnxruntime-web/wasm`
 *
 * Published npm package always loads via package imports (not CDN).
 * GitHub Pages demo may rewrite these imports to CDN at build time.
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
    return isWebAssemblyJspiAvailable() ? 'jspi' : 'webgpu';
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

  // Native WebGPU / JSPI bundles also include wasm EP.
  if ((loaded === 'webgpu' || loaded === 'jspi') && requested === 'wasm') {
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
    case 'jspi':
      return import('onnxruntime-web/jspi');
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
 * Lazily loads the matching npm package entry from `executionProviders` / `ortBundle`.
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

const WEBGPU_SESSION_BUSY = /another WebGPU EP inference session is being created/i;
let webGpuSessionLock: Promise<void> = Promise.resolve();

function sessionUsesWebGpu(options?: OrtTypes.InferenceSession.SessionOptions): boolean {
  const providers = options?.executionProviders ?? [];
  return providers.some(provider => {
    if (typeof provider === 'string') {
      return provider === 'webgpu';
    }
    return (provider as { name?: string }).name === 'webgpu';
  });
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

function createInferenceSession(
  model: string | ArrayBufferLike | Uint8Array,
  options?: OrtTypes.InferenceSession.SessionOptions,
): Promise<OrtTypes.InferenceSession> {
  if (typeof model === 'string') {
    return ort.InferenceSession.create(model, options);
  }
  if (model instanceof Uint8Array) {
    return ort.InferenceSession.create(model, options);
  }
  return ort.InferenceSession.create(model, options);
}

/**
 * onnxruntime-web WebGPU EP only allows one InferenceSession.create at a time.
 * Queue WebGPU session creation globally so SAM3's multiple encoders/decoders cannot race.
 */
export async function createOrtInferenceSession(
  model: string | ArrayBufferLike | Uint8Array,
  options?: OrtTypes.InferenceSession.SessionOptions,
): Promise<OrtTypes.InferenceSession> {
  if (!sessionUsesWebGpu(options)) {
    return createInferenceSession(model, options);
  }

  let release!: () => void;
  const previous = webGpuSessionLock;
  webGpuSessionLock = new Promise<void>(resolve => {
    release = resolve;
  });
  await previous;

  try {
    let lastError: unknown;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        return await createInferenceSession(model, options);
      } catch (error) {
        lastError = error;
        const message = error instanceof Error ? error.message : String(error);
        if (!WEBGPU_SESSION_BUSY.test(message) || attempt === 5) {
          throw error;
        }
        await delay(40 * (attempt + 1));
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  } finally {
    release();
  }
}

/** Compatible alias used after initialization (Tensor / InferenceSession / env). */
export const ort: OrtModule = new Proxy({} as OrtModule, {
  get(_target, property, receiver) {
    return Reflect.get(getOrt(), property, receiver);
  },
}) as OrtModule;

export type { OrtTypes };
