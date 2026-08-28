import type * as OrtTypes from 'onnxruntime-web';
import { DrawTool } from './draw-tool';
import { Sam3Handler } from './handler/sam3/sam3-handler';
import { ClipBpeTokenizer, loadClipTokenizer } from './handler/sam3/clip-tokenizer';
import { SAM3_IMAGE_SIZE, SAM3_MASK_SIZE, SAM3_TEXT_LENGTH } from './handler/sam3/preprocess';
import type {
  Sam3BoxPrompt,
  Sam3ImageInput,
  Sam3InferenceState,
  Sam3MaskPrompt,
  Sam3Options,
  Sam3PcsPrompt,
  Sam3PointPrompt,
  Sam3PvsPrompt,
  Sam3PvsResult,
} from './handler/sam3/types';
import { ensureOnnxRuntimeWebInitialized, ort } from './runtime';
import type { Point, Rect, Segmentation, SegmentationDrawingOptions, YoloModelSource } from './types';

const DEFAULT_EXECUTION_PROVIDERS = ['wasm'] as const;
type Sam3SessionKind = 'vision' | 'text' | 'grounding' | 'prompt';

function usesWebGpu(options: Sam3Options): boolean {
  const providers = options.sessionOptions?.executionProviders ?? options.executionProviders ?? DEFAULT_EXECUTION_PROVIDERS;
  return providers.some(provider => {
    if (typeof provider === 'string') {
      return provider === 'webgpu';
    }

    return (provider as { name?: string }).name === 'webgpu';
  });
}

export class Sam3 {
  private visionSession: OrtTypes.InferenceSession | null = null;
  private textSession: OrtTypes.InferenceSession | null = null;
  private groundingSession: OrtTypes.InferenceSession | null = null;
  private promptSession: OrtTypes.InferenceSession | null = null;
  private handler: Sam3Handler | null = null;
  private tokenizer: ClipBpeTokenizer | null = null;
  private state: Sam3InferenceState | null = null;
  private readonly webGpu: boolean;

  constructor(private readonly options: Sam3Options) {
    this.webGpu = usesWebGpu(options);
  }

  static async create(options: Sam3Options): Promise<Sam3> {
    const sam3 = new Sam3(options);
    await sam3.load();
    return sam3;
  }

  get isLoaded(): boolean {
    return this.handler !== null;
  }

  get inferenceState(): Sam3InferenceState | null {
    return this.state;
  }

  async load(): Promise<this> {
    const defaultNumThreads = this.webGpu ? 0 : 1;
    await ensureOnnxRuntimeWebInitialized({
      ...this.options,
      numThreads: this.options.numThreads ?? defaultNumThreads,
    });
    if (ort.env.wasm) {
      ort.env.wasm.numThreads = this.options.numThreads ?? defaultNumThreads;
      ort.env.wasm.proxy = this.options.proxy ?? false;
      ort.env.wasm.simd = true;
    }

    if (this.webGpu) {
      const webgpu = (ort.env as { webgpu?: { powerPreference?: 'low-power' | 'high-performance' } }).webgpu;
      if (webgpu && webgpu.powerPreference === undefined) {
        webgpu.powerPreference = 'high-performance';
      }
    }

    await this.dispose();

    try {
      const progress = this.options.onLoadProgress;
      progress?.('正在加载视觉编码器...');
      this.visionSession = await this.createSession('vision', this.options.visionEncoder);

      progress?.('正在加载文本 / Grounding / Prompt / tokenizer...');
      const promptSource = this.options.promptDecoder;
      const textPromise = this.createSession('text', this.options.textEncoder);
      const groundingPromise = this.createSession('grounding', this.options.groundingDecoder);
      const promptPromise = promptSource ? this.createSession('prompt', promptSource) : Promise.resolve(null);
      const tokenizerPromise = this.options.tokenizer
        ? loadClipTokenizer(this.options.tokenizer)
        : Promise.resolve(null);

      try {
        const [textSession, groundingSession, promptSession, tokenizer] = await Promise.all([
          textPromise,
          groundingPromise,
          promptPromise,
          tokenizerPromise,
        ]);
        this.textSession = textSession;
        this.groundingSession = groundingSession;
        this.promptSession = promptSession;
        this.tokenizer = tokenizer;
      } catch (error) {
        const settled = await Promise.allSettled([textPromise, groundingPromise, promptPromise]);
        await Promise.all(
          settled.map(item =>
            item.status === 'fulfilled' && item.value ? item.value.release() : Promise.resolve(),
          ),
        );
        throw error;
      }

      if (!this.tokenizer) {
        throw new Error('SAM3 tokenizer tables are required. Pass tokenizer: clip_bpe.json or parsed tables.');
      }

      this.handler = new Sam3Handler(
        this.visionSession,
        this.textSession,
        this.groundingSession,
        this.promptSession,
        this.tokenizer,
        {
          confidenceThreshold: this.options.confidenceThreshold ?? 0.5,
          pixelConfidence: this.options.pixelConfidence ?? 0.5,
          imageSize: this.options.imageSize ?? SAM3_IMAGE_SIZE,
          maskSize: this.options.maskSize ?? SAM3_MASK_SIZE,
          webGpu: this.webGpu,
        },
      );
      return this;
    } catch (error) {
      await this.dispose();
      throw error;
    }
  }

  async setImage(image: Sam3ImageInput): Promise<Sam3InferenceState> {
    this.state = await this.ensureHandler().setImage(image, this.state);
    return this.state;
  }

  async setTextPrompt(prompt: string, description?: string): Promise<Segmentation[]> {
    return this.ensureHandler().setTextPrompt(prompt, this.ensureState(), description);
  }

  async addGeometricPrompt(box: Rect, label = true): Promise<Segmentation[]> {
    return this.ensureHandler().addGeometricPrompt(box, label, this.ensureState());
  }

  async addGeometricPoint(point: Point, label = true): Promise<Segmentation[]> {
    return this.ensureHandler().addGeometricPoint(point, label, this.ensureState());
  }

  async removeGeometricPrompt(index: number): Promise<Segmentation[]> {
    return this.ensureHandler().removeGeometricPrompt(index, this.ensureState());
  }

  async removeGeometricPoint(index: number): Promise<Segmentation[]> {
    return this.ensureHandler().removeGeometricPoint(index, this.ensureState());
  }

  resetPrompts(): Sam3InferenceState {
    return this.ensureHandler().resetPrompts(this.ensureState());
  }

  resetVisualPrompts(): Sam3InferenceState {
    return this.ensureHandler().resetVisualPrompts(this.ensureState());
  }

  async setConfidenceThreshold(threshold: number): Promise<Segmentation[] | Sam3InferenceState> {
    this.ensureHandler().setConfidenceThreshold(threshold);
    const state = this.ensureState();
    if (state.lastPcsRaw?.length || state.textEmbeddings) {
      return this.ensureHandler().applyConfidenceThreshold(state);
    }
    return state;
  }

  async predictConcept(prompt: Sam3PcsPrompt): Promise<Segmentation[]> {
    return this.ensureHandler().predictConcept(prompt, this.ensureState());
  }

  async predictVisual(prompt: Sam3PvsPrompt = {}): Promise<Sam3PvsResult> {
    return this.ensureHandler().predictVisual(prompt, this.ensureState());
  }

  async addPoint(point: Point, label: 0 | 1 = 1): Promise<Sam3PvsResult> {
    return this.ensureHandler().addPoint(point, label, this.ensureState());
  }

  async addBox(box: Rect): Promise<Sam3PvsResult> {
    return this.ensureHandler().addBox(box, this.ensureState());
  }

  async addMask(mask: Sam3MaskPrompt): Promise<Sam3PvsResult> {
    return this.ensureHandler().addMask(mask, this.ensureState());
  }

  selectVisualCandidate(index: number): Sam3PvsResult {
    return this.ensureHandler().selectPvsCandidate(index, this.ensureState());
  }

  async removePoint(index: number): Promise<Sam3PvsResult> {
    return this.ensureHandler().removePoint(index, this.ensureState());
  }

  drawSegmentations(
    source: Sam3ImageInput,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    DrawTool.drawSegmentationEdgePoints(source, segmentations, canvas, {
      drawBoundingBoxes: true,
      drawLabel: true,
      drawSegmentationPixelMask: true,
      fillSegmentationEdgePoints: true,
      ...options,
    });
  }

  drawSegmentationEdgePoints(
    source: Sam3ImageInput,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    DrawTool.drawSegmentationEdgePoints(source, segmentations, canvas, options);
  }

  async dispose(): Promise<void> {
    this.handler?.dispose();
    const sessions = [this.visionSession, this.textSession, this.groundingSession, this.promptSession];
    this.visionSession = null;
    this.textSession = null;
    this.groundingSession = null;
    this.promptSession = null;
    this.handler = null;
    this.tokenizer = null;
    this.state = null;
    await Promise.all(sessions.filter(Boolean).map(session => session?.release()));
  }

  private createSessionOptions(kind: Sam3SessionKind): OrtTypes.InferenceSession.SessionOptions {
    const userOptions = this.options.sessionOptions ?? {};
    const userExtra = (userOptions.extra ?? {}) as Record<string, unknown>;
    const userSession = (userExtra.session ?? {}) as Record<string, unknown>;
    const keepVisionOnGpu = this.webGpu && kind === 'vision';
    const disableGraphOpt = kind === 'vision' || kind === 'text';
    const options: OrtTypes.InferenceSession.SessionOptions = {
      enableCpuMemArena: true,
      enableMemPattern: true,
      ...userOptions,
      graphOptimizationLevel: userOptions.graphOptimizationLevel ?? (disableGraphOpt ? 'disabled' : 'all'),
      extra: {
        ...userExtra,
        session: {
          strict_shape_type_inference: '0',
          ...(keepVisionOnGpu ? { use_device_allocator_for_initializers: '1' } : {}),
          ...userSession,
        },
      },
      executionProviders: userOptions.executionProviders ?? this.createExecutionProviders(),
    };

    if (keepVisionOnGpu && options.preferredOutputLocation == null) {
      options.preferredOutputLocation = 'gpu-buffer';
    }

    return options;
  }

  private createExecutionProviders(): OrtTypes.InferenceSession.SessionOptions['executionProviders'] {
    const providers = [...(this.options.executionProviders ?? DEFAULT_EXECUTION_PROVIDERS)];
    if (!this.webGpu) {
      return providers as OrtTypes.InferenceSession.SessionOptions['executionProviders'];
    }

    return providers.map(provider => {
      if (provider === 'webgpu') {
        return { name: 'webgpu', preferredLayout: 'NCHW', validationMode: 'wgpuOnly' };
      }

      return provider;
    }) as OrtTypes.InferenceSession.SessionOptions['executionProviders'];
  }

  private async createSession(kind: Sam3SessionKind, model: YoloModelSource): Promise<OrtTypes.InferenceSession> {
    const label = `${kind}-encoder`;
    const options = this.createSessionOptions(kind);

    try {
      return await this.createSessionWithOptions(model, options);
    } catch (error) {
      if (kind === 'vision' && this.webGpu && options.preferredOutputLocation === 'gpu-buffer') {
        try {
          return await this.createSessionWithOptions(model, { ...options, preferredOutputLocation: undefined });
        } catch {
          // fall through
        }
      }

      if (options.graphOptimizationLevel === 'all') {
        try {
          return await this.createSessionWithOptions(model, { ...options, graphOptimizationLevel: 'disabled' });
        } catch {
          // fall through to the original error
        }
      }

      const message = error instanceof Error ? error.message : String(error);
      if (/bad_alloc/i.test(message)) {
        throw new Error(
          `Failed to create SAM3 session "${label}": 浏览器 WASM 内存不足。请加载 fp16 模型（vision-encoder.fp16.onnx / text-encoder.fp16.onnx），并优先使用 WebGPU。原始错误: ${message}`,
        );
      }
      throw new Error(`Failed to create SAM3 session "${label}": ${message}`);
    }
  }

  private async createSessionWithOptions(
    model: YoloModelSource,
    options: OrtTypes.InferenceSession.SessionOptions,
  ): Promise<OrtTypes.InferenceSession> {
    if (typeof model === 'string') {
      return ort.InferenceSession.create(model, options);
    }

    if (model instanceof Uint8Array) {
      return ort.InferenceSession.create(model, options);
    }

    return ort.InferenceSession.create(model, options);
  }

  private ensureHandler(): Sam3Handler {
    if (!this.handler) {
      throw new Error('SAM3 is not loaded. Call Sam3.create() / load() first.');
    }

    return this.handler;
  }

  private ensureState(): Sam3InferenceState {
    if (!this.state) {
      throw new Error('You must call setImage() before prompting SAM3.');
    }

    return this.state;
  }
}

export type {
  Sam3BoxPrompt,
  Sam3ImageInput,
  Sam3InferenceState,
  Sam3MaskPrompt,
  Sam3Options,
  Sam3PcsPrompt,
  Sam3PcsRawOutput,
  Sam3PointPrompt,
  Sam3PvsPrompt,
  Sam3PvsResult,
  Sam3TokenizerTables,
} from './handler/sam3/types';
export { SAM3_IMAGE_SIZE, SAM3_MASK_SIZE, SAM3_TEXT_LENGTH };
