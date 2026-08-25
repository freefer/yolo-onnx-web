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

export class Sam3 {
  private visionSession: OrtTypes.InferenceSession | null = null;
  private textSession: OrtTypes.InferenceSession | null = null;
  private groundingSession: OrtTypes.InferenceSession | null = null;
  private promptSession: OrtTypes.InferenceSession | null = null;
  private handler: Sam3Handler | null = null;
  private tokenizer: ClipBpeTokenizer | null = null;
  private state: Sam3InferenceState | null = null;

  constructor(private readonly options: Sam3Options) {}

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
    await ensureOnnxRuntimeWebInitialized({
      ...this.options,
      numThreads: this.options.numThreads ?? 1,
    });
    await this.dispose();

    const sessionOptions = this.createSessionOptions();
    const progress = this.options.onLoadProgress;
    progress?.('正在加载视觉编码器...');
    this.visionSession = await this.createSession(this.options.visionEncoder, sessionOptions, 'vision-encoder');
    progress?.('正在加载文本编码器...');
    this.textSession = await this.createSession(this.options.textEncoder, sessionOptions, 'text-encoder');
    progress?.('正在加载概念分割解码器...');
    this.groundingSession = await this.createSession(this.options.groundingDecoder, sessionOptions, 'grounding-decoder');
    if (this.options.promptDecoder) {
      progress?.('正在加载视觉分割解码器...');
      this.promptSession = await this.createSession(this.options.promptDecoder, sessionOptions, 'prompt-decoder');
    } else {
      this.promptSession = null;
    }
    progress?.('正在加载 CLIP tokenizer...');
    this.tokenizer = this.options.tokenizer
      ? await loadClipTokenizer(this.options.tokenizer)
      : null;

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
      },
    );
    return this;
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
    if (state.textEmbeddings) {
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

  private createSessionOptions(): OrtTypes.InferenceSession.SessionOptions {
    const userOptions = this.options.sessionOptions ?? {};
    const userExtra = (userOptions.extra ?? {}) as Record<string, unknown>;
    const userSession = (userExtra.session ?? {}) as Record<string, unknown>;

    return {
      enableCpuMemArena: false,
      enableMemPattern: false,
      ...userOptions,
      graphOptimizationLevel: userOptions.graphOptimizationLevel ?? 'disabled',
      extra: {
        ...userExtra,
        session: {
          strict_shape_type_inference: '0',
          ...userSession,
        },
      },
      executionProviders:
        userOptions.executionProviders ??
        ([...(this.options.executionProviders ?? DEFAULT_EXECUTION_PROVIDERS)] as OrtTypes.InferenceSession.SessionOptions['executionProviders']),
    };
  }

  private async createSession(
    model: YoloModelSource,
    options: OrtTypes.InferenceSession.SessionOptions,
    label: string,
  ): Promise<OrtTypes.InferenceSession> {
    try {
      if (typeof model === 'string') {
        return await ort.InferenceSession.create(model, options);
      }

      if (model instanceof Uint8Array) {
        return await ort.InferenceSession.create(model, options);
      }

      return await ort.InferenceSession.create(model, options);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/bad_alloc/i.test(message)) {
        throw new Error(
          `Failed to create SAM3 session "${label}": 浏览器 WASM 内存不足。请加载 fp16 模型（vision-encoder.fp16.onnx / text-encoder.fp16.onnx），并优先使用 WebGPU。原始错误: ${message}`,
        );
      }
      throw new Error(`Failed to create SAM3 session "${label}": ${message}`);
    }
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
  Sam3PointPrompt,
  Sam3PvsPrompt,
  Sam3PvsResult,
  Sam3TokenizerTables,
} from './handler/sam3/types';
export { SAM3_IMAGE_SIZE, SAM3_MASK_SIZE, SAM3_TEXT_LENGTH };
