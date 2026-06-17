import type * as OrtTypes from 'onnxruntime-web';
import { parseOnnxModel } from './onnx-model';
import { ensureOnnxRuntimeWebInitialized, ort } from './runtime';
import type {
  Classification,
  ClassificationDrawingOptions,
  DetectionDrawingOptions,
  IYoloHandler,
  KeyPoint,
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

const DEFAULT_EXECUTION_PROVIDERS = ['wasm'] as const;
const DEFAULT_BOX_COLORS = [
  '#22c55e',
  '#3b82f6',
  '#f97316',
  '#e11d48',
  '#8b5cf6',
  '#14b8a6',
  '#f59e0b',
  '#06b6d4',
] as const;
const DEFAULT_POSE_CONNECTIONS: readonly [number, number][] = [
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
];
 
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
    this._onnxModel = await parseOnnxModel(this.session, model);

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
    const { drawWidth, drawHeight, xPad, yPad, gain } = this.calculateProportionalResize(
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

    for (let i = 0, pixel = 0; i < imageData.length; i += 4, pixel += 1) {
      tensorData[pixel] = imageData[i] / 255;
      tensorData[pixelCount + pixel] = imageData[i + 1] / 255;
      tensorData[pixelCount * 2 + pixel] = imageData[i + 2] / 255;
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
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    this.drawBoundingBoxes(context, detections, width, height, options);
  }

  drawClassifications(
    source: YoloImageSource,
    classifications: readonly Classification[],
    canvas: HTMLCanvasElement,
    options: ClassificationDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const fontColor = options.fontColor ?? '#f8fafc';
    const backgroundColor = options.backgroundColor ?? 'rgba(15, 23, 42, 0.72)';
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;
    const margin = 10;
    const lineGap = 8;

    context.font = font;
    context.textBaseline = 'top';

    const lineHeight = this.getCanvasFontSize(font) + lineGap;
    const labels = classifications.map(item => `${item.label}${drawConfidenceScore ? ` (${(item.confidence * 100).toFixed(1)}%)` : ''}`);
    const boxWidth = Math.max(0, ...labels.map(label => context.measureText(label).width)) + margin * 2;
    const boxHeight = labels.length * lineHeight + margin * 2 - lineGap;

    if (drawLabelBackground && labels.length > 0) {
      context.fillStyle = backgroundColor;
      context.fillRect(8, 8, boxWidth, boxHeight);
    }

    context.fillStyle = fontColor;

    for (let i = 0; i < labels.length; i += 1) {
      context.fillText(labels[i], 8 + margin, 8 + margin + i * lineHeight);
    }
  }

  drawObbDetections(
    source: YoloImageSource,
    detections: readonly OBBDetection[],
    canvas: HTMLCanvasElement,
    options: DetectionDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 320));
    const drawLabel = options.drawLabel ?? true;
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];

    context.font = font;
    context.textBaseline = 'middle';
    context.lineWidth = lineWidth;

    for (const detection of detections) {
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, options.boundingBoxOpacity);
      const points = this.getObbCorners(detection.boundingBox, detection.orientationAngle);

      context.strokeStyle = color;
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        context.lineTo(points[i].x, points[i].y);
      }
      context.closePath();
      context.stroke();

      if (drawLabel) {
        this.drawDetectionLabel(context, detection, points[2].x, points[2].y, color, {
          font,
          drawConfidenceScore,
          drawLabelBackground,
          fontColor: options.fontColor,
        });
      }
    }
  }

  drawSegmentations(
    source: YoloImageSource,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const drawMask = options.drawSegmentationPixelMask ?? true;
    const drawContour = options.drawContour ?? false;
    const drawBoundingBoxes = options.drawBoundingBoxes ?? true;
    const pixelMaskOpacity = options.pixelMaskOpacity ?? 128;

    if (drawMask) {
      for (const segmentation of segmentations) {
        this.drawSegmentationMask(context, segmentation, this.getDetectionColor(segmentation, colors, undefined, pixelMaskOpacity));
      }
    }

    if (drawContour) {
      for (const segmentation of segmentations) {
        this.drawSegmentationContour(
          context,
          segmentation,
          this.getDetectionColor(segmentation, colors, options.strokeStyle),
          options.contourThickness ?? 2,
        );
      }
    }

    if (drawBoundingBoxes || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, segmentations, width, height, options);
    }
  }

  drawPoseEstimations(
    source: YoloImageSource,
    poseEstimations: readonly PoseEstimation[],
    canvas: HTMLCanvasElement,
    options: PoseDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const confidence = options.poseConfidence ?? 0.25;
    const defaultPoseColor = options.defaultPoseColor ?? '#22c55e';
    const radius = options.keyPointRadius ?? Math.max(3, Math.round(Math.min(width, height) / 260));
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 360));
    const markers = options.keyPointMarkers;

    context.lineWidth = lineWidth;

    for (const pose of poseEstimations) {
      this.drawPoseConnections(context, pose.keyPoints, confidence, markers, defaultPoseColor);

      for (let i = 0; i < pose.keyPoints.length; i += 1) {
        const keyPoint = pose.keyPoints[i];

        if (keyPoint.confidence < confidence) {
          continue;
        }

        context.fillStyle = markers?.[i]?.color ?? defaultPoseColor;
        context.beginPath();
        context.arc(keyPoint.x, keyPoint.y, radius, 0, Math.PI * 2);
        context.fill();
      }
    }

    if ((options.drawBoundingBoxes ?? true) || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, poseEstimations, width, height, options);
    }
  }

  private drawBoundingBoxes(
    context: CanvasRenderingContext2D,
    detections: readonly ObjectDetection[],
    width: number,
    height: number,
    options: DetectionDrawingOptions = {},
  ): void {
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 320));
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const drawLabel = options.drawLabel ?? true;
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;

    context.lineWidth = lineWidth;
    context.font = font;
    context.textBaseline = 'middle';

    for (const detection of detections) {
      const { left, top, right, bottom } = detection.boundingBox;
      const boxWidth = right - left;
      const boxHeight = bottom - top;
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, options.boundingBoxOpacity);

      if (boxWidth <= 0 || boxHeight <= 0) {
        continue;
      }

      context.strokeStyle = color;
      context.strokeRect(left, top, boxWidth, boxHeight);

      if (drawLabel) {
        this.drawDetectionLabel(context, detection, left, Math.max(0, top - this.getCanvasFontSize(font)), color, {
          font,
          drawConfidenceScore,
          drawLabelBackground,
          fontColor: options.fontColor,
        });
      }
    }
  }

  private prepareDrawingCanvas(
    source: YoloImageSource,
    canvas: HTMLCanvasElement,
    drawSource = true,
  ): { context: CanvasRenderingContext2D; width: number; height: number } {
    const { width, height } = this.getImageSourceSize(source);
    const context = canvas.getContext('2d');

    if (!context) {
      throw new Error('Canvas 2D context is not available.');
    }

    if (canvas.width !== width) {
      canvas.width = width;
    }

    if (canvas.height !== height) {
      canvas.height = height;
    }

    context.clearRect(0, 0, width, height);

    if (drawSource) {
      context.drawImage(source, 0, 0, width, height);
    }

    return { context, width, height };
  }

  private getDetectionColor(
    detection: ObjectDetection,
    colors: readonly string[],
    fallback?: string,
    alpha = 255,
  ): string {
    const color = fallback ?? colors[detection.label.index % colors.length] ?? DEFAULT_BOX_COLORS[0];
    return this.withAlpha(color, alpha);
  }

  private withAlpha(color: string, alpha: number): string {
    if (!color.startsWith('#') || color.length !== 7) {
      return color;
    }

    const r = Number.parseInt(color.slice(1, 3), 16);
    const g = Number.parseInt(color.slice(3, 5), 16);
    const b = Number.parseInt(color.slice(5, 7), 16);
    const normalizedAlpha = this.clamp(alpha, 0, 255) / 255;

    return `rgba(${r}, ${g}, ${b}, ${normalizedAlpha})`;
  }

  private drawDetectionLabel(
    context: CanvasRenderingContext2D,
    detection: ObjectDetection,
    x: number,
    y: number,
    backgroundColor: string,
    options: {
      font: string;
      drawConfidenceScore: boolean;
      drawLabelBackground: boolean;
      fontColor?: string;
    },
  ): void {
    const fontSize = this.getCanvasFontSize(options.font);
    const margin = Math.max(4, Math.round(fontSize / 3));
    const label = `${detection.label.name}${options.drawConfidenceScore ? ` ${(detection.confidence * 100).toFixed(1)}%` : ''}`;
    const textWidth = context.measureText(label).width;
    const boxWidth = textWidth + margin * 2;
    const boxHeight = fontSize + margin * 2;
    const left = this.clamp(Math.round(x), 0, Math.max(0, context.canvas.width - boxWidth));
    const top = this.clamp(Math.round(y), 0, Math.max(0, context.canvas.height - boxHeight));

    context.font = options.font;
    context.textBaseline = 'middle';

    if (options.drawLabelBackground) {
      context.fillStyle = backgroundColor;
      context.fillRect(left, top, boxWidth, boxHeight);
    }

    context.fillStyle = options.fontColor ?? '#f8fafc';
    context.fillText(label, left + margin, top + boxHeight / 2);
  }

  private getCanvasFontSize(font: string): number {
    const match = font.match(/(\d+(?:\.\d+)?)px/);
    return match ? Number(match[1]) : 14;
  }

  private getObbCorners(box: Rect, radians: number): { x: number; y: number }[] {
    const centerX = (box.left + box.right) / 2;
    const centerY = (box.top + box.bottom) / 2;
    const cos = Math.cos(radians);
    const sin = Math.sin(radians);
    const corners = [
      { x: box.left, y: box.top },
      { x: box.right, y: box.top },
      { x: box.right, y: box.bottom },
      { x: box.left, y: box.bottom },
    ];

    return corners.map(point => {
      const dx = point.x - centerX;
      const dy = point.y - centerY;

      return {
        x: centerX + dx * cos - dy * sin,
        y: centerY + dx * sin + dy * cos,
      };
    });
  }

  private drawSegmentationMask(context: CanvasRenderingContext2D, segmentation: Segmentation, color: string): void {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }

    const imageData = context.createImageData(width, height);
    const rgba = this.parseCanvasColor(color);
    const maskCanvas = document.createElement('canvas');
    const maskContext = maskCanvas.getContext('2d');

    if (!maskContext) {
      throw new Error('Canvas 2D context is not available.');
    }

    maskCanvas.width = width;
    maskCanvas.height = height;

    for (let pixelIndex = 0; pixelIndex < width * height; pixelIndex += 1) {
      if (!this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex)) {
        continue;
      }

      const offset = pixelIndex * 4;
      imageData.data[offset] = rgba.r;
      imageData.data[offset + 1] = rgba.g;
      imageData.data[offset + 2] = rgba.b;
      imageData.data[offset + 3] = rgba.a;
    }

    maskContext.putImageData(imageData, 0, 0);
    context.drawImage(maskCanvas, left, top);
  }

  private drawSegmentationContour(
    context: CanvasRenderingContext2D,
    segmentation: Segmentation,
    color: string,
    thickness: number,
  ): void {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }

    context.fillStyle = color;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixelIndex = y * width + x;

        if (!this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex)) {
          continue;
        }

        const isEdge =
          x === 0 ||
          x === width - 1 ||
          y === 0 ||
          y === height - 1 ||
          !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex - 1) ||
          !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex + 1) ||
          !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex - width) ||
          !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex + width);

        if (isEdge) {
          context.fillRect(left + x, top + y, thickness, thickness);
        }
      }
    }
  }

  private isPackedMaskSet(mask: Uint8Array, pixelIndex: number): boolean {
    if (pixelIndex < 0) {
      return false;
    }

    const byteIndex = pixelIndex >> 3;

    if (byteIndex >= mask.byteLength) {
      return false;
    }

    return (mask[byteIndex] & (1 << (pixelIndex & 0b0111))) !== 0;
  }

  private parseCanvasColor(color: string): { r: number; g: number; b: number; a: number } {
    if (color.startsWith('#') && color.length === 7) {
      return {
        r: Number.parseInt(color.slice(1, 3), 16),
        g: Number.parseInt(color.slice(3, 5), 16),
        b: Number.parseInt(color.slice(5, 7), 16),
        a: 255,
      };
    }

    const rgba = color.match(/rgba?\(([^)]+)\)/);

    if (rgba) {
      const parts = rgba[1].split(',').map(part => Number(part.trim()));

      return {
        r: parts[0] ?? 34,
        g: parts[1] ?? 197,
        b: parts[2] ?? 94,
        a: Math.round((parts[3] ?? 1) * 255),
      };
    }

    return { r: 34, g: 197, b: 94, a: 128 };
  }

  private drawPoseConnections(
    context: CanvasRenderingContext2D,
    keyPoints: readonly KeyPoint[],
    confidence: number,
    markers: PoseDrawingOptions['keyPointMarkers'],
    defaultColor: string,
  ): void {
    if (markers && markers.length > 0) {
      for (let sourceIndex = 0; sourceIndex < markers.length; sourceIndex += 1) {
        const source = keyPoints[sourceIndex];

        if (!source || source.confidence < confidence) {
          continue;
        }

        for (const connection of markers[sourceIndex]?.connections ?? []) {
          this.drawPoseConnection(context, source, keyPoints[connection.index], confidence, connection.color ?? defaultColor);
        }
      }

      return;
    }

    for (const [sourceIndex, targetIndex] of DEFAULT_POSE_CONNECTIONS) {
      this.drawPoseConnection(context, keyPoints[sourceIndex], keyPoints[targetIndex], confidence, defaultColor);
    }
  }

  private drawPoseConnection(
    context: CanvasRenderingContext2D,
    source: KeyPoint | undefined,
    target: KeyPoint | undefined,
    confidence: number,
    color: string,
  ): void {
    if (!source || !target || source.confidence < confidence || target.confidence < confidence) {
      return;
    }

    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(source.x, source.y);
    context.lineTo(target.x, target.y);
    context.stroke();
  }

  tensor<T extends OrtTypes.Tensor.Type>(
    type: T,
    data: OrtTypes.Tensor.DataTypeMap[T],
    dims?: readonly number[],
  ): OrtTypes.Tensor {
    return new ort.Tensor(type, data, dims);
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
      WORLDV2: ['ObjectDetection'],
    };

    return supportMap[modelVersion].includes(modelType);
  }
}
