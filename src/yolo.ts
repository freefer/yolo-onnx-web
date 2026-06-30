import type * as OrtTypes from 'onnxruntime-web';
import { DrawTool } from './draw-tool';
import { parseOnnxModel } from './onnx-model';
import { ensureOnnxRuntimeWebInitialized, ort } from './runtime';
import type {
  Classification,
  ClassificationDrawingOptions,
  DetectionDrawingOptions,
  IYoloHandler,
  ModelType,
  ModelVersion,
  OBBDetection,
  ObjectDetection,
  OnnxModel,
  PoseDrawingOptions,
  PoseEstimation,
  Rect,
  Segmentation,
  SegmentationDrawingOptions,
  YoloPreprocessResult,
  YoloFeeds,
  YoloFetches,
  YoloImageSource,
  YoloModelSource,
  YoloOptions,
  YoloRunOptions,
  YoloRunResult,
} from './types';
import { Yolo26Handler } from './handler/yolo26/yolo26-hanlder';
import { Yolov10Handler } from './handler/yolov10/yolov10-handler';
import { Yolov8Handler } from './handler/yolov8/yolov8-handler';
import { RT_DETRHandler } from './handler/RT-DETR/RT-DETR-handler';
import { RF_DETRHandler } from './handler/RF-DETR/RF-DETR-handler';

const DEFAULT_EXECUTION_PROVIDERS = ['wasm'] as const;
 
export class Yolo {
  private readonly options: YoloOptions;
  private readonly model?: YoloModelSource;
  private session: OrtTypes.InferenceSession | null = null;
  private _onnxModel: OnnxModel | null = null;
  private _handler: IYoloHandler | null = null;
  private preprocessCanvas: HTMLCanvasElement | null = null;
  private preprocessContext: CanvasRenderingContext2D | null = null;
  private preprocessTensorData: Float32Array | null = null;
  private preprocessTensorSize = 0;

  constructor(options: YoloOptions = {}) {
    this.options = options;
    this.model = options.model;
    ensureOnnxRuntimeWebInitialized(options);
  }

  get yoloOptions(): YoloOptions {
    return this.options;
  }

  get preprocessBackend(): 'cpu' | 'webgpu' {
    const requestedBackend = this.options.preprocessBackend ?? (this.hasWebGpuExecutionProvider() ? 'webgpu' : 'cpu');

    if (requestedBackend !== 'webgpu') {
      return 'cpu';
    }

    return this.hasWebGpuExecutionProvider() ? 'webgpu' : 'cpu';
  }

  static async create(options: YoloOptions): Promise<Yolo> {
    const yolo = new Yolo(options);

    if (options.model) {
      await yolo.load(options.model);
    }

    return yolo;
  }

  get isLoaded(): boolean {
    return this.session !== null;
  }

  get inputNames(): readonly string[] {
    return this.ensureSession().inputNames;
  }

  get outputNames(): readonly string[] {
    return this.ensureSession().outputNames;
  }

  get onnxModel(): OnnxModel {
    if (!this._onnxModel) {
      throw new Error('ONNX model info is not parsed. Call load() first or pass model to Yolo.create().');
    }

    return this._onnxModel;
  }

  async load(model: YoloModelSource = this.requireModel()): Promise<this> {
    await this.dispose();

    this.session = await this.createSession(model);

    // 解析 ONNX 模型信息，结构对齐 YoloDotNet 的 OnnxModel。
    this._onnxModel = await parseOnnxModel(this.session, model, this.options);

    var modelVersion = this._onnxModel.modelVersion;
    var modelType = this._onnxModel.modelType;

    if (!this.isSupportedModel(modelVersion, modelType)) {
      throw new Error(`Unsupported model type ${modelType} for model version ${modelVersion}.`);
    }

    switch (modelVersion) {
      case 'V5U':
      case 'V8':
      case 'V8E':
      case 'V9':
      case 'V11':
      case 'V11E':
      case 'V12':
      case 'WORLDV2':
        this._handler = new Yolov8Handler(this);
        break;
      case 'V10':
        this._handler = new Yolov10Handler(this);
        break;
      case 'V26':
        this._handler = new Yolo26Handler(this);
        break;
      case 'RTDETR':
        this._handler = new RT_DETRHandler(this);
        break;
      case 'RFDETR':
        this._handler = new RF_DETRHandler(this);
        break;
      default:
        throw new Error('Unsupported model version: ' + modelVersion);
    }

    return this;
  }

  run(feeds: YoloFeeds, options?: YoloRunOptions): Promise<YoloRunResult> {
    return this.ensureSession().run(feeds, options);
  }

  runWithFetches(feeds: YoloFeeds, fetches: YoloFetches, options?: YoloRunOptions): Promise<YoloRunResult> {
    return this.ensureSession().run(feeds, fetches, options);
  }

  predict(feeds: YoloFeeds, options?: YoloRunOptions): Promise<YoloRunResult> {
    return this.run(feeds, options);
  }

  RunObjectDetection(
    img: YoloImageSource,
    confidence: number = 0.2,
    iou: number = 0.7,
    roi: Rect | null = null,
  ): Promise<ObjectDetection[]> {
    return this.ensureHandler().RunObjectDetection(img, confidence, iou, roi);
  }

  RunObbDetection(
    img: YoloImageSource,
    confidence: number = 0.2,
    iou: number = 0.7,
    roi: Rect | null = null,
  ): Promise<OBBDetection[]> {
    return this.ensureHandler().RunObbDetection(img, confidence, iou, roi);
  }

  RunSegmentation(
    img: YoloImageSource,
    confidence: number = 0.2,
    pixelConfidence: number = 0.65,
    iou: number = 0.7,
    roi: Rect | null = null,
  ): Promise<Segmentation[]> {
    return this.ensureHandler().RunSegmentation(img, confidence, pixelConfidence, iou, roi);
  }

  RunPoseEstimation(
    img: YoloImageSource,
    confidence: number = 0.2,
    iou: number = 0.7,
    roi: Rect | null = null,
  ): Promise<PoseEstimation[]> {
    return this.ensureHandler().RunPoseEstimation(img, confidence, iou, roi);
  }

  RunClassification(img: YoloImageSource, classes: number = 5): Promise<Classification[]> {
    return this.ensureHandler().RunClassification(img, classes);
  }

  preprocessImage(img: YoloImageSource, roi: Rect | null = null): YoloPreprocessResult {
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;

    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = this.options.imageResize ?? 'proportional';
    const { drawWidth, drawHeight, xPad, yPad, gain } =
      resizeMode === 'stretch'
        ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 }
        : this.calculateProportionalResize(
            sourceRect.width,
            sourceRect.height,
            modelWidth,
            modelHeight,
          );
    const context = this.getPreprocessContext(modelWidth, modelHeight);

    context.clearRect(0, 0, modelWidth, modelHeight);
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      xPad,
      yPad,
      drawWidth,
      drawHeight,
    );

    const imageData = context.getImageData(0, 0, modelWidth, modelHeight).data;
    const pixelCount = modelWidth * modelHeight;
    const tensorData = this.getPreprocessTensorData(channels * pixelCount);
    const imageMean = this.options.imageMean;
    const imageStd = this.options.imageStd;

    for (let i = 0, pixel = 0; i < imageData.length; i += 4, pixel += 1) {
      const r = imageData[i] / 255;
      const g = imageData[i + 1] / 255;
      const b = imageData[i + 2] / 255;

      if (imageMean && imageStd) {
        tensorData[pixel] = (r - imageMean[0]) / imageStd[0];
        tensorData[pixelCount + pixel] = (g - imageMean[1]) / imageStd[1];
        tensorData[pixelCount * 2 + pixel] = (b - imageMean[2]) / imageStd[2];
      } else {
        tensorData[pixel] = r;
        tensorData[pixelCount + pixel] = g;
        tensorData[pixelCount * 2 + pixel] = b;
      }
    }

    return {
      tensorData,
      inputName: this.inputNames[0],
      inputShape,
      sourceWidth: sourceRect.width,
      sourceHeight: sourceRect.height,
      xPad,
      yPad,
      gain,
      resizeMode,
      roi,
    };
  }

  scaleBoundingBox(x1: number, y1: number, x2: number, y2: number, input: YoloPreprocessResult): Rect {
    const roiLeft = input.roi?.left ?? 0;
    const roiTop = input.roi?.top ?? 0;

    return {
      left: roiLeft + this.clamp(Math.trunc((x1 - input.xPad) * input.gain), 0, input.sourceWidth - 1),
      top: roiTop + this.clamp(Math.trunc((y1 - input.yPad) * input.gain), 0, input.sourceHeight - 1),
      right: roiLeft + this.clamp(Math.trunc((x2 - input.xPad) * input.gain), 0, input.sourceWidth),
      bottom: roiTop + this.clamp(Math.trunc((y2 - input.yPad) * input.gain), 0, input.sourceHeight),
    };
  }

  drawObjectDetections(
    source: YoloImageSource,
    detections: readonly ObjectDetection[],
    canvas: HTMLCanvasElement,
    options: DetectionDrawingOptions = {},
  ): void {
    DrawTool.drawObjectDetections(source, detections, canvas, options);
  }

  drawClassifications(
    source: YoloImageSource,
    classifications: readonly Classification[],
    canvas: HTMLCanvasElement,
    options: ClassificationDrawingOptions = {},
  ): void {
    DrawTool.drawClassifications(source, classifications, canvas, options);
  }

  drawObbDetections(
    source: YoloImageSource,
    detections: readonly OBBDetection[],
    canvas: HTMLCanvasElement,
    options: DetectionDrawingOptions = {},
  ): void {
    DrawTool.drawObbDetections(source, detections, canvas, options);
  }

  drawSegmentations(
    source: YoloImageSource,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    DrawTool.drawSegmentations(source, segmentations, canvas, options);
  }

  drawPoseEstimations(
    source: YoloImageSource,
    poseEstimations: readonly PoseEstimation[],
    canvas: HTMLCanvasElement,
    options: PoseDrawingOptions = {},
  ): void {
    DrawTool.drawPoseEstimations(source, poseEstimations, canvas, options);
  }

  extractSegmentationEdgePoints(segmentation: Segmentation): { x: number; y: number }[] {
    return DrawTool.extractSegmentationEdgePoints(segmentation);
  }

  extractSegmentationsEdgePoints(segmentations: readonly Segmentation[]): { x: number; y: number }[][] {
    return DrawTool.extractSegmentationsEdgePoints(segmentations);
  }

  tensor<T extends OrtTypes.Tensor.Type>(
    type: T,
    data: OrtTypes.Tensor.DataTypeMap[T],
    dims?: readonly number[],
  ): OrtTypes.Tensor {
    return new ort.Tensor(type, data, dims);
  }

  async getWebGpuDevice(): Promise<any> {
    const device = (ort.env as any).webgpu?.device;

    if (!device) {
      throw new Error('WebGPU device is not initialized by ONNX Runtime Web.');
    }

    return device;
  }

  tensorFromGpuBuffer(
    gpuBuffer: OrtTypes.Tensor.GpuBufferType,
    dims: readonly number[],
    dispose?: () => void,
  ): OrtTypes.Tensor {
    return ort.Tensor.fromGpuBuffer(gpuBuffer, {
      dataType: 'float32',
      dims,
      dispose,
    });
  }

  async dispose(): Promise<void> {
    if (!this.session) {
      return;
    }

    await this.session.release();
    this.session = null;
    this._onnxModel = null;
    this._handler = null;
    this.preprocessCanvas = null;
    this.preprocessContext = null;
    this.preprocessTensorData = null;
    this.preprocessTensorSize = 0;
  }

  private createSessionOptions(): OrtTypes.InferenceSession.SessionOptions {
    return {
      graphOptimizationLevel: 'all',
      ...this.options.sessionOptions,
      executionProviders:
        this.options.sessionOptions?.executionProviders ??
        ([...(this.options.executionProviders ?? DEFAULT_EXECUTION_PROVIDERS)] as OrtTypes.InferenceSession.SessionOptions['executionProviders']),
    };
  }

  private createSession(model: YoloModelSource): Promise<OrtTypes.InferenceSession> {
    const options: OrtTypes.InferenceSession.SessionOptions = this.createSessionOptions();
 
    if (typeof model === 'string') {

      return ort.InferenceSession.create(model, options);
    }

    if (model instanceof Uint8Array) {
      return ort.InferenceSession.create(model, options);
    }

    return ort.InferenceSession.create(model, options);
  }

  private ensureSession(): OrtTypes.InferenceSession {
    if (!this.session) {
      throw new Error('Yolo model is not loaded. Call load() first or pass model to Yolo.create().');
    }

    return this.session;
  }

  private getInputShape(): readonly [number, number, number, number] {
    const shape = Object.values(this.onnxModel.inputShapes)[0];

    if (!shape || shape.length !== 4) {
      throw new Error(`Unsupported YOLO input shape: ${JSON.stringify(shape)}`);
    }

    return [shape[0], shape[1], shape[2], shape[3]] as const;
  }

  private getSourceRect(img: YoloImageSource, roi: Rect | null): { x: number; y: number; width: number; height: number } {
    const { width, height } = this.getImageSourceSize(img);

    if (!roi) {
      return { x: 0, y: 0, width, height };
    }

    const left = this.clamp(Math.trunc(roi.left), 0, width - 1);
    const top = this.clamp(Math.trunc(roi.top), 0, height - 1);
    const right = this.clamp(Math.trunc(roi.right), left + 1, width);
    const bottom = this.clamp(Math.trunc(roi.bottom), top + 1, height);

    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    };
  }

  private calculateProportionalResize(
    sourceWidth: number,
    sourceHeight: number,
    modelWidth: number,
    modelHeight: number,
  ): { drawWidth: number; drawHeight: number; xPad: number; yPad: number; gain: number } {
    if (sourceWidth < modelWidth && sourceHeight < modelHeight) {
      return {
        drawWidth: sourceWidth,
        drawHeight: sourceHeight,
        xPad: (modelWidth - sourceWidth) * 0.5,
        yPad: (modelHeight - sourceHeight) * 0.5,
        gain: 1,
      };
    }

    const ratio = Math.min(modelWidth / sourceWidth, modelHeight / sourceHeight);
    const drawWidth = sourceWidth * ratio;
    const drawHeight = sourceHeight * ratio;

    return {
      drawWidth,
      drawHeight,
      xPad: (modelWidth - drawWidth) * 0.5,
      yPad: (modelHeight - drawHeight) * 0.5,
      gain: Math.max(sourceWidth / modelWidth, sourceHeight / modelHeight),
    };
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  private getPreprocessContext(width: number, height: number): CanvasRenderingContext2D {
    if (!this.preprocessCanvas) {
      this.preprocessCanvas = document.createElement('canvas');
    }

    if (this.preprocessCanvas.width !== width) {
      this.preprocessCanvas.width = width;
    }

    if (this.preprocessCanvas.height !== height) {
      this.preprocessCanvas.height = height;
    }

    if (!this.preprocessContext) {
      this.preprocessContext = this.preprocessCanvas.getContext('2d', { willReadFrequently: true });
    }

    if (!this.preprocessContext) {
      throw new Error('Canvas 2D context is not available.');
    }

    return this.preprocessContext;
  }

  private getPreprocessTensorData(size: number): Float32Array {
    if (!this.preprocessTensorData || this.preprocessTensorSize !== size) {
      this.preprocessTensorData = new Float32Array(size);
      this.preprocessTensorSize = size;
    }

    return this.preprocessTensorData;
  }

  private getImageSourceSize(img: YoloImageSource): { width: number; height: number } {
    if (img instanceof HTMLImageElement) {
      return {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
      };
    }

    if (img instanceof HTMLVideoElement) {
      return {
        width: img.videoWidth || img.width,
        height: img.videoHeight || img.height,
      };
    }

    if ('displayWidth' in img && 'displayHeight' in img) {
      return {
        width: img.displayWidth || img.codedWidth,
        height: img.displayHeight || img.codedHeight,
      };
    }

    if (img instanceof SVGImageElement) {
      const width = img.width.baseVal.value || img.getBoundingClientRect().width;
      const height = img.height.baseVal.value || img.getBoundingClientRect().height;

      return { width, height };
    }

    return {
      width: img.width,
      height: img.height,
    };
  }

  private ensureHandler(): IYoloHandler {
    if (!this._handler) {
      throw new Error('YOLO handler is not initialized. Call load() first or pass model to Yolo.create().');
    }

    return this._handler;
  }

  private requireModel(): YoloModelSource {
    if (!this.model) {
      throw new Error('Missing model source. Pass model in constructor options or load(model).');
    }

    return this.model;
  }

  private hasWebGpuExecutionProvider(): boolean {
    const executionProviders =
      this.options.sessionOptions?.executionProviders ??
      this.options.executionProviders ??
      DEFAULT_EXECUTION_PROVIDERS;

    return executionProviders.some(executionProvider => {
      if (typeof executionProvider === 'string') {
        return executionProvider === 'webgpu';
      }

      return (executionProvider as any)?.name === 'webgpu';
    });
  }

  private isSupportedModel(modelVersion: ModelVersion, modelType: ModelType): boolean {
    const allTasks: readonly ModelType[] = [
      'Classification',
      'ObjectDetection',
      'ObbDetection',
      'Segmentation',
      'PoseEstimation',
    ];

    const supportMap: Record<ModelVersion, readonly ModelType[]> = {
      V5U: ['ObjectDetection'],
      V8: allTasks,
      V8E: ['Segmentation'],
      V9: ['ObjectDetection'],
      V10: ['ObjectDetection'],
      V11: allTasks,
      V11E: ['Segmentation'],
      V12: allTasks,
      V26: allTasks,
      RTDETR: ['ObjectDetection'],
      RFDETR: ['ObjectDetection', 'Segmentation'],
      WORLDV2: ['ObjectDetection'],
    };

    return supportMap[modelVersion].includes(modelType);
  }
}
