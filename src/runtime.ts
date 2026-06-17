import * as ort from 'onnxruntime-web/all';
import type { OnnxRuntimeWebOptions } from './types';

let configured = false;

/**
 * Configure onnxruntime-web before creating an inference session.
 */
export function initializeOnnxRuntimeWeb(options: OnnxRuntimeWebOptions = {}): void {
  if (options.wasmPaths !== undefined) {
    ort.env.wasm.wasmPaths = options.wasmPaths as typeof ort.env.wasm.wasmPaths;
  }

  if (options.numThreads !== undefined) {
    ort.env.wasm.numThreads = options.numThreads;
  }

  if (options.proxy !== undefined) {
    ort.env.wasm.proxy = options.proxy;
  }

  configured = true;
}

export function ensureOnnxRuntimeWebInitialized(options: OnnxRuntimeWebOptions = {}): void {
  if (configured) {
    initializeOnnxRuntimeWeb(options);
    return;
  }

  initializeOnnxRuntimeWeb(options);
}

export { ort };
