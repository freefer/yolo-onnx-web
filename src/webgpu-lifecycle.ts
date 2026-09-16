export type WebGpuDeviceLostInfo = {
  reason?: string;
  message?: string;
};

export type WebGpuLifecycleListener = (info?: WebGpuDeviceLostInfo) => void | Promise<void>;

const DEVICE_LOST_RE =
  /device lost|GPUDevice was lost|parent device is lost|GPUBuffer used after destroy|Destroyed (?:buffer|texture)|Device is lost|Device was destroyed|device\.destroy|GPU process crashed|Invalid CommandBuffer|vk::Device|DXGI_ERROR_DEVICE_REMOVED|GPUDevice\.lost/i;

type GpuDeviceLike = {
  lost: Promise<WebGpuDeviceLostInfo>;
  createBuffer: (descriptor: { size: number; usage: number }) => { destroy: () => void };
};

type GpuAdapterLike = {
  requestDevice: (descriptor?: unknown) => Promise<GpuDeviceLike>;
};

type OrtWebGpuBackend = {
  device?: GpuDeviceLike;
  env?: { webgpu?: { device?: unknown } };
  initialize: (env: unknown, adapter: GpuAdapterLike) => Promise<void>;
  dispose: () => void;
};

type OrtLike = {
  env?: {
    webgpu?: Record<string, unknown> & {
      powerPreference?: 'low-power' | 'high-performance';
    };
  };
};

let getOrtFn: (() => OrtLike) | null = null;
let capturedBackend: OrtWebGpuBackend | null = null;
let watchedDevice: GpuDeviceLike | null = null;
let deviceLost = false;
let deviceGeneration = 0;
let recoveryPromise: Promise<void> | null = null;
let recoveryDepth = 0;
let sessionTeardownDepth = 0;
let wasmHooksInstalled = false;
let pageHooksInstalled = false;

const lostListeners = new Set<WebGpuLifecycleListener>();
const restoredListeners = new Set<WebGpuLifecycleListener>();

export function bindOrtGetter(getter: () => unknown): void {
  getOrtFn = getter as () => OrtLike;
}

export function isWebGpuDeviceLostError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error);
  return DEVICE_LOST_RE.test(message);
}

export function isOrtWebGpuDeviceLost(): boolean {
  return deviceLost;
}

export function getOrtWebGpuDeviceGeneration(): number {
  return deviceGeneration;
}

export function onOrtWebGpuDeviceLost(listener: WebGpuLifecycleListener): () => void {
  lostListeners.add(listener);
  return () => lostListeners.delete(listener);
}

export function onOrtWebGpuDeviceRestored(listener: WebGpuLifecycleListener): () => void {
  restoredListeners.add(listener);
  return () => restoredListeners.delete(listener);
}

export function installWebGpuLifecycleHooks(): void {
  installWasmJsepCapture();
  installPageLifecycleHooks();
}

/**
 * Session release on the JSPI WebGPU EP often calls GPUDevice.destroy().
 * Mark that window so device.lost(reason=destroyed) is not treated as a sleep/GPU crash.
 */
export async function withOrtWebGpuSessionTeardown<T>(operation: () => Promise<T>): Promise<T> {
  sessionTeardownDepth += 1;
  try {
    return await operation();
  } finally {
    await Promise.resolve();
    sessionTeardownDepth -= 1;
  }
}

export async function watchOrtWebGpuDevice(): Promise<void> {
  installWebGpuLifecycleHooks();
  const device = await resolveOrtWebGpuDevice();
  if (!device || device === watchedDevice) {
    return;
  }
  watchDevice(device);
}

export async function ensureOrtWebGpuReady(): Promise<void> {
  if (recoveryPromise && recoveryDepth === 0) {
    await recoveryPromise;
  }
  if (deviceLost && recoveryDepth === 0) {
    await recoverOrtWebGpuDevice();
  }
}

export async function recoverOrtWebGpuDevice(info?: WebGpuDeviceLostInfo): Promise<void> {
  if (recoveryDepth > 0) {
    return;
  }
  if (recoveryPromise) {
    return recoveryPromise;
  }
  if (isExpectedGpuDeviceDestroy(info) && !deviceLost) {
    watchedDevice = null;
    return;
  }

  recoveryPromise = (async () => {
    recoveryDepth += 1;
    deviceLost = true;
    const lostInfo = info ?? { reason: 'unknown', message: 'WebGPU device was released' };
    let notifiedLost = false;
    try {
      await notify(lostListeners, lostInfo);
      notifiedLost = true;
      await waitUntilDocumentVisible();
      await delay(50);
      const rebuilt = await reinitializeOrtWebGpuBackend(lostInfo);
      deviceLost = false;
      if (rebuilt) {
        deviceGeneration += 1;
      }
      await notify(restoredListeners, lostInfo);
      if (rebuilt) {
        await watchOrtWebGpuDevice();
      }
    } catch (error) {
      deviceLost = false;
      deviceGeneration += 1;
      if (notifiedLost) {
        await notify(restoredListeners, lostInfo);
      }
      console.warn('[yolo-onnx-web] WebGPU 恢复失败，将在下次创建会话时重试:', error);
    } finally {
      recoveryDepth -= 1;
    }
  })().finally(() => {
    recoveryPromise = null;
  });

  return recoveryPromise;
}

function isExpectedGpuDeviceDestroy(info?: WebGpuDeviceLostInfo): boolean {
  const reason = info?.reason ?? '';
  const message = info?.message ?? '';
  const destroyed = reason === 'destroyed' || /device was destroyed/i.test(message);
  if (!destroyed) {
    return false;
  }
  return sessionTeardownDepth > 0;
}

function captureWebGpuBackend(impl: unknown): void {
  if (!impl || typeof impl !== 'object') {
    return;
  }
  const backend = impl as OrtWebGpuBackend;
  if (typeof backend.initialize === 'function' && typeof backend.dispose === 'function') {
    capturedBackend = backend;
  }
}

function wrapJsepInit(fn: Function): Function {
  if ((fn as { __yoloOrtWrapped?: boolean }).__yoloOrtWrapped) {
    return fn;
  }

  const wrapped = function jsepInit(this: unknown, name: string, impl?: unknown[]) {
    if (name === 'webgpu' && Array.isArray(impl) && impl[0]) {
      captureWebGpuBackend(impl[0]);
    }
    return fn.apply(this, arguments as unknown as []);
  };
  (wrapped as { __yoloOrtWrapped?: boolean }).__yoloOrtWrapped = true;
  return wrapped;
}

function installWasmJsepCapture(): void {
  if (wasmHooksInstalled || typeof WebAssembly === 'undefined') {
    return;
  }
  wasmHooksInstalled = true;

  const wrapInstance = (instance: WebAssembly.Instance): WebAssembly.Instance => {
    const exportsObject = instance.exports as WebAssembly.Exports & { jsepInit?: Function };
    if (typeof exportsObject.jsepInit !== 'function') {
      return instance;
    }

    return new Proxy(instance, {
      get(target, property, receiver) {
        if (property !== 'exports') {
          return Reflect.get(target, property, receiver);
        }

        return new Proxy(exportsObject, {
          get(exportTarget, exportName, exportReceiver) {
            const value = Reflect.get(exportTarget, exportName, exportReceiver);
            if (exportName !== 'jsepInit' || typeof value !== 'function') {
              return value;
            }
            return wrapJsepInit(value);
          },
        });
      },
    });
  };

  const originalInstantiate = WebAssembly.instantiate.bind(WebAssembly);
  WebAssembly.instantiate = (async (
    source: Parameters<typeof WebAssembly.instantiate>[0],
    imports?: WebAssembly.Imports,
  ) => {
    const result = await originalInstantiate(source as never, imports);
    if ('instance' in result) {
      return { ...result, instance: wrapInstance(result.instance) };
    }
    return wrapInstance(result as WebAssembly.Instance) as Awaited<ReturnType<typeof WebAssembly.instantiate>>;
  }) as typeof WebAssembly.instantiate;

  if (typeof WebAssembly.instantiateStreaming === 'function') {
    const originalStreaming = WebAssembly.instantiateStreaming.bind(WebAssembly);
    WebAssembly.instantiateStreaming = (async (source, imports) => {
      const result = await originalStreaming(source, imports);
      return { ...result, instance: wrapInstance(result.instance) };
    }) as typeof WebAssembly.instantiateStreaming;
  }
}

function installPageLifecycleHooks(): void {
  if (pageHooksInstalled || typeof window === 'undefined' || typeof document === 'undefined') {
    return;
  }
  pageHooksInstalled = true;

  const probe = () => {
    void probeWebGpuDevice();
  };

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      probe();
    }
  });
  window.addEventListener('pageshow', probe);
}

async function probeWebGpuDevice(): Promise<void> {
  if (recoveryPromise || sessionTeardownDepth > 0) {
    return;
  }

  const device = watchedDevice ?? (await resolveOrtWebGpuDevice().catch(() => null));
  if (!device) {
    return;
  }

  if (await isDeviceLostSettled(device)) {
    const info = await device.lost.catch(() => ({ reason: 'unknown', message: 'WebGPU device was lost' }));
    if (info?.reason === 'destroyed' || isExpectedGpuDeviceDestroy(info)) {
      watchedDevice = null;
      deviceLost = false;
      return;
    }
    await recoverOrtWebGpuDevice({
      reason: info?.reason || 'unknown',
      message: info?.message || 'WebGPU device was lost while the page was in the background',
    }).catch(error => {
      console.warn('[yolo-onnx-web] WebGPU 后台恢复失败:', error);
    });
    return;
  }

  try {
    const usage = (globalThis as unknown as { GPUBufferUsage?: { COPY_DST: number; MAP_READ: number } }).GPUBufferUsage;
    if (!usage) {
      return;
    }
    const buffer = device.createBuffer({ size: 16, usage: usage.COPY_DST | usage.MAP_READ });
    buffer.destroy();
  } catch (error) {
    if (isWebGpuDeviceLostError(error)) {
      await recoverOrtWebGpuDevice({
        reason: 'unknown',
        message: error instanceof Error ? error.message : String(error),
      }).catch(recoverError => {
        console.warn('[yolo-onnx-web] WebGPU 探测恢复失败:', recoverError);
      });
    }
  }
}

function watchDevice(device: GpuDeviceLike): void {
  watchedDevice = device;
  void device.lost.then((info: WebGpuDeviceLostInfo) => {
    if (watchedDevice !== device) {
      return;
    }
    watchedDevice = null;
    if (isExpectedGpuDeviceDestroy(info) || info?.reason === 'destroyed') {
      deviceLost = false;
      return;
    }
    void recoverOrtWebGpuDevice({
      reason: info?.reason,
      message: info?.message || 'WebGPU device.lost',
    }).catch(error => {
      console.warn('[yolo-onnx-web] WebGPU device.lost 恢复失败:', error);
    });
  });
}

async function reinitializeOrtWebGpuBackend(info: WebGpuDeviceLostInfo): Promise<boolean> {
  let ort: OrtLike | null = null;
  try {
    ort = getOrtFn?.() ?? null;
  } catch {
    ort = null;
  }
  const webgpu = ort?.env?.webgpu as { device?: unknown; powerPreference?: 'low-power' | 'high-performance' } | undefined;
  if (webgpu) {
    try {
      delete (webgpu as { device?: unknown }).device;
    } catch {
      // ignore
    }
  }

  if (!capturedBackend || typeof capturedBackend.initialize !== 'function') {
    // JSPI WebGPU EP owns the device in wasm and has no JS WebGpuBackend.initialize().
    // Session release often destroys the GPUDevice; the next InferenceSession.create
    // requests a new one. Do not treat that as a fatal, unrecoverable error.
    console.warn(
      `[yolo-onnx-web] WebGPU 设备已释放（${info.message || info.reason || 'device lost'}）。未捕获 JSEP 后端，跳过原地重建。`,
    );
    return false;
  }

  try {
    capturedBackend.dispose();
  } catch {
    // buffers/textures on a lost device often throw while destroying
  }

  const adapter = await requestWebGpuAdapter();
  await capturedBackend.initialize(ort?.env, adapter);
  return true;
}

async function requestWebGpuAdapter(): Promise<GpuAdapterLike> {
  const gpu = (
    navigator as Omit<Navigator, 'gpu'> & {
      gpu?: {
        requestAdapter: (options?: { powerPreference?: string }) => Promise<GpuAdapterLike | null>;
      };
    }
  ).gpu;
  if (!gpu) {
    throw new Error('当前浏览器不支持 WebGPU');
  }

  const powerPreference =
    (getOrtFn?.()?.env?.webgpu as { powerPreference?: 'low-power' | 'high-performance' } | undefined)?.powerPreference ??
    'high-performance';
  let lastError: unknown;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    try {
      const adapter = await gpu.requestAdapter({ powerPreference });
      if (adapter) {
        return adapter;
      }
    } catch (error) {
      lastError = error;
    }
    await delay(120 * (attempt + 1));
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('无法重新申请 WebGPU 适配器，请刷新页面后重试');
}

async function resolveOrtWebGpuDevice(): Promise<GpuDeviceLike | null> {
  try {
    const ort = getOrtFn?.();
    const device = await ((ort?.env?.webgpu as { device?: Promise<GpuDeviceLike> | GpuDeviceLike } | undefined)?.device);
    return device ?? capturedBackend?.device ?? null;
  } catch {
    return capturedBackend?.device ?? null;
  }
}

function isDeviceLostSettled(device: GpuDeviceLike): Promise<boolean> {
  return new Promise(resolve => {
    let settled = false;
    const finish = (lost: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(lost);
    };
    void device.lost.then(() => finish(true));
    queueMicrotask(() => finish(false));
  });
}

async function waitUntilDocumentVisible(): Promise<void> {
  if (typeof document === 'undefined' || document.visibilityState === 'visible') {
    return;
  }

  await new Promise<void>(resolve => {
    const onChange = () => {
      if (document.visibilityState === 'visible') {
        document.removeEventListener('visibilitychange', onChange);
        resolve();
      }
    };
    document.addEventListener('visibilitychange', onChange);
  });
}

async function notify(listeners: Set<WebGpuLifecycleListener>, info: WebGpuDeviceLostInfo): Promise<void> {
  for (const listener of [...listeners]) {
    try {
      await listener(info);
    } catch {
      // listeners must not abort recovery
    }
  }
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

installWebGpuLifecycleHooks();
