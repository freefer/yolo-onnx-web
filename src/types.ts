import type * as ort from 'onnxruntime-web';

export type YoloModelSource = string | ArrayBufferLike | Uint8Array;
export type YoloImageSource = CanvasImageSource;

export type YoloExecutionProvider = NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>[number];
export const YoloExecutionProviderNames = [
  'coreml',
  'cpu',
  'cuda',
  'dml',
  'nnapi',
  'tensorrt',
  'wasm',
  'webgl',
  'webgpu',
  'webnn',
  'qnn',
  'xnnpack',
] as const satisfies readonly YoloExecutionProvider[];

export const YoloExecutionProviderOptions = [
  { value: 'coreml', label: 'CoreML' },
  { value: 'cpu', label: 'CPU' },
  { value: 'cuda', label: 'CUDA' },
  { value: 'dml', label: 'DirectML' },
  { value: 'nnapi', label: 'NNAPI' },
  { value: 'tensorrt', label: 'TensorRT' },
  { value: 'wasm', label: 'WASM' },
  { value: 'webgl', label: 'WebGL' },
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'webnn', label: 'WebNN' },
  { value: 'qnn', label: 'QNN' },
  { value: 'xnnpack', label: 'XNNPACK' },
] as const satisfies readonly { value: YoloExecutionProvider; label: string }[];

export const YoloWebExecutionProviderOptions = [
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'wasm', label: 'WASM' },
 
  { value: 'webnn', label: 'WebNN' },
  { value: 'webgl', label: 'WebGL' },
  { value: 'cpu', label: 'CPU' },
] as const satisfies readonly { value: YoloExecutionProvider; label: string }[];

export type ModelType = 'Classification' | 'ObjectDetection' | 'ObbDetection' | 'Segmentation' | 'PoseEstimation';

export type ModelVersion = 'V5U' | 'V8' | 'V8E' | 'V9' | 'V10' | 'V11' | 'V11E' | 'V12' | 'V26' | 'RTDETR' | 'WORLDV2';

export type ModelDataType = 'Float' | 'Float16';

export interface LabelModel {
  index: number;
  name: string;
}

export interface Point {
  x: number;
  y: number;
}

export interface KeyPoint extends Point {
  confidence: number;
}

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface Detection {
  label: LabelModel;
  confidence: number;
  boundingBox: Rect;
  id?: number;
  tail?: Point[];
}

export interface DetectionDrawingOptions {
  strokeStyle?: string;
  fillStyle?: string;
  lineWidth?: number;
  font?: string;
  drawLabel?: boolean;
  drawSource?: boolean;
  drawConfidenceScore?: boolean;
  drawLabelBackground?: boolean;
  boundingBoxHexColors?: string[];
  boundingBoxOpacity?: number;
  fontColor?: string;
}

export interface ClassificationDrawingOptions {
  font?: string;
  fontColor?: string;
  backgroundColor?: string;
  drawLabelBackground?: boolean;
  drawSource?: boolean;
  drawConfidenceScore?: boolean;
}

export interface SegmentationDrawingOptions extends DetectionDrawingOptions {
  drawSegmentationPixelMask?: boolean;
  pixelMaskOpacity?: number;
  drawContour?: boolean;
  contourThickness?: number;
  drawBoundingBoxes?: boolean;
}

export interface KeyPointConnection {
  index: number;
  color?: string;
}

export interface KeyPointMarker {
  color?: string;
  connections?: KeyPointConnection[];
}

export interface PoseDrawingOptions extends DetectionDrawingOptions {
  poseConfidence?: number;
  defaultPoseColor?: string;
  keyPointMarkers?: KeyPointMarker[];
  keyPointRadius?: number;
  drawBoundingBoxes?: boolean;
}

export interface YoloPreprocessResult {
  tensorData: Float32Array;
  inputName: string;
  inputShape: readonly [number, number, number, number];
  sourceWidth: number;
  sourceHeight: number;
  xPad: number;
  yPad: number;
  gain: number;
  roi: Rect | null;
}

export class TrackingInfo {
  id?: number;
  tail?: Point[];

  constructor(options: Pick<TrackingInfo, 'id' | 'tail'> = {}) {
    this.id = options.id;
    this.tail = options.tail;
  }
}

export class ObjectDetection extends TrackingInfo implements Detection {
  label: LabelModel;
  confidence: number;
  boundingBox: Rect;

  constructor(options: Detection) {
    super({ id: options.id, tail: options.tail });
    this.label = options.label;
    this.confidence = options.confidence;
    this.boundingBox = options.boundingBox;
  }
}

export class OBBDetection extends ObjectDetection {
  orientationAngle: number;

  constructor(options: Detection & { orientationAngle: number }) {
    super(options);
    this.orientationAngle = options.orientationAngle;
  }
}

export class Segmentation extends ObjectDetection {
  bitPackedPixelMask: Uint8Array;

  constructor(options: Detection & { bitPackedPixelMask: Uint8Array }) {
    super(options);
    this.bitPackedPixelMask = options.bitPackedPixelMask;
  }
}

export class PoseEstimation extends ObjectDetection {
  keyPoints: KeyPoint[];

  constructor(options: Detection & { keyPoints: KeyPoint[] }) {
    super(options);
    this.keyPoints = options.keyPoints;
  }
}

export class Classification
{
  label: string;
  confidence: number;

  constructor(label: string, confidence: number) {
    this.label = label;
    this.confidence = confidence;
  }
}

export interface OnnxModel {
  modelType: ModelType;
  modelVersion: ModelVersion;
  modelDataType: ModelDataType;
  inputShapes: Record<string, number[]>;
  outputShapes: Record<string, number[]>;
  labels: LabelModel[];
  inputShapeSize: number;
  customMetaData: Record<string, string>;
}

export interface OnnxRuntimeWebOptions {
  /**
   * Prefix or mapping used by onnxruntime-web to locate its wasm files.
   * Example: /examples/browser/ort-wasm/
   */
  wasmPaths?: string | Record<string, string>;

  /** Number of wasm worker threads. Keep undefined to use onnxruntime-web defaults. */
  numThreads?: number;

  /** Whether to run wasm backend in a proxy worker. */
  proxy?: boolean;
}

export interface YoloOptions extends OnnxRuntimeWebOptions {
  /** ONNX model URL, ArrayBuffer, or Uint8Array. */
  model?: YoloModelSource;

  /** Browser execution provider priority. Defaults to ['wasm']. */
  executionProviders?: readonly YoloExecutionProvider[];

  /** Extra onnxruntime-web session options. */
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

export type YoloFeeds = ort.InferenceSession.FeedsType;
export type YoloFetches = ort.InferenceSession.FetchesType;
export type YoloRunOptions = ort.InferenceSession.RunOptions;
export type YoloRunResult = ort.InferenceSession.ReturnType;
export type YoloTensor = ort.Tensor;

export interface IYoloHandler{
 
  preprocessImage(img: YoloImageSource, roi?: Rect | null): YoloPreprocessResult
  RunObjectDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<ObjectDetection[]>
  RunObbDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<OBBDetection[]>
  RunSegmentation(img: YoloImageSource, confidence: number, pixelConfidence: number, iou: number, roi?: Rect | null): Promise<Segmentation[]>
  RunPoseEstimation(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<PoseEstimation[]>
  RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]>

}
