import * as ort from 'onnxruntime-web';
import * as all from 'onnxruntime-web/all';
export { all as ort };

type YoloModelSource = string | ArrayBufferLike | Uint8Array;
type YoloImageSource = CanvasImageSource;
type YoloExecutionProvider = NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>[number];
declare const YoloExecutionProviderNames: readonly ["coreml", "cpu", "cuda", "dml", "nnapi", "tensorrt", "wasm", "webgl", "webgpu", "webnn", "qnn", "xnnpack"];
declare const YoloExecutionProviderOptions: readonly [{
    readonly value: "coreml";
    readonly label: "CoreML";
}, {
    readonly value: "cpu";
    readonly label: "CPU";
}, {
    readonly value: "cuda";
    readonly label: "CUDA";
}, {
    readonly value: "dml";
    readonly label: "DirectML";
}, {
    readonly value: "nnapi";
    readonly label: "NNAPI";
}, {
    readonly value: "tensorrt";
    readonly label: "TensorRT";
}, {
    readonly value: "wasm";
    readonly label: "WASM";
}, {
    readonly value: "webgl";
    readonly label: "WebGL";
}, {
    readonly value: "webgpu";
    readonly label: "WebGPU";
}, {
    readonly value: "webnn";
    readonly label: "WebNN";
}, {
    readonly value: "qnn";
    readonly label: "QNN";
}, {
    readonly value: "xnnpack";
    readonly label: "XNNPACK";
}];
declare const YoloWebExecutionProviderOptions: readonly [{
    readonly value: "webgpu";
    readonly label: "WebGPU";
}, {
    readonly value: "wasm";
    readonly label: "WASM";
}, {
    readonly value: "webnn";
    readonly label: "WebNN";
}, {
    readonly value: "webgl";
    readonly label: "WebGL";
}, {
    readonly value: "cpu";
    readonly label: "CPU";
}];
type ModelType = 'Classification' | 'ObjectDetection' | 'ObbDetection' | 'Segmentation' | 'PoseEstimation';
type ModelVersion = 'V5U' | 'V8' | 'V8E' | 'V9' | 'V10' | 'V11' | 'V11E' | 'V12' | 'V26' | 'RTDETR' | 'RFDETR' | 'WORLDV2';
type ModelDataType = 'Float' | 'Float16';
interface LabelModel {
    index: number;
    name: string;
}
type YoloLabels = readonly string[] | string;
interface Point {
    x: number;
    y: number;
}
interface KeyPoint extends Point {
    confidence: number;
}
interface Rect {
    left: number;
    top: number;
    right: number;
    bottom: number;
}
interface Detection {
    label: LabelModel;
    confidence: number;
    boundingBox: Rect;
    id?: number;
    tail?: Point[];
}
interface DetectionDrawingOptions {
    strokeStyle?: string;
    fillStyle?: string;
    lineWidth?: number;
    font?: string;
    drawLabel?: boolean;
    drawSource?: boolean;
    drawConfidenceScore?: boolean;
    drawLabelBackground?: boolean;
    boundingBoxHexColors?: string[];
    resultOpacity?: number;
    boundingBoxOpacity?: number;
    fontColor?: string;
}
interface ClassificationDrawingOptions {
    font?: string;
    fontColor?: string;
    backgroundColor?: string;
    drawLabelBackground?: boolean;
    drawSource?: boolean;
    drawConfidenceScore?: boolean;
}
interface SegmentationDrawingOptions extends DetectionDrawingOptions {
    drawSegmentationPixelMask?: boolean;
    pixelMaskOpacity?: number;
    drawContour?: boolean;
    contourThickness?: number;
    drawBoundingBoxes?: boolean;
    segmentationEdgePoints?: readonly (readonly Point[])[];
    fillSegmentationEdgePoints?: boolean;
}
interface KeyPointConnection {
    index: number;
    color?: string;
}
interface KeyPointMarker {
    color?: string;
    connections?: KeyPointConnection[];
}
interface PoseDrawingOptions extends DetectionDrawingOptions {
    poseConfidence?: number;
    defaultPoseColor?: string;
    keyPointMarkers?: KeyPointMarker[];
    keyPointRadius?: number;
    drawBoundingBoxes?: boolean;
}
interface YoloPreprocessResult {
    tensorData: Float32Array;
    inputTensor?: YoloTensor;
    inputName: string;
    inputShape: readonly [number, number, number, number];
    sourceWidth: number;
    sourceHeight: number;
    xPad: number;
    yPad: number;
    gain: number;
    resizeMode: 'proportional' | 'stretch';
    roi: Rect | null;
}
declare class TrackingInfo {
    id?: number;
    tail?: Point[];
    constructor(options?: Pick<TrackingInfo, 'id' | 'tail'>);
}
declare class ObjectDetection extends TrackingInfo implements Detection {
    label: LabelModel;
    confidence: number;
    boundingBox: Rect;
    constructor(options: Detection);
}
declare class OBBDetection extends ObjectDetection {
    orientationAngle: number;
    constructor(options: Detection & {
        orientationAngle: number;
    });
}
declare class Segmentation extends ObjectDetection {
    bitPackedPixelMask: Uint8Array;
    segmentationEdgePoints?: Point[];
    constructor(options: Detection & {
        bitPackedPixelMask: Uint8Array;
        segmentationEdgePoints?: Point[];
    });
}
declare class PoseEstimation extends ObjectDetection {
    keyPoints: KeyPoint[];
    constructor(options: Detection & {
        keyPoints: KeyPoint[];
    });
}
declare class Classification {
    label: string;
    confidence: number;
    constructor(label: string, confidence: number);
}
interface OnnxModel {
    modelType: ModelType;
    modelVersion: ModelVersion;
    modelDataType: ModelDataType;
    inputShapes: Record<string, number[]>;
    outputShapes: Record<string, number[]>;
    labels: LabelModel[];
    inputShapeSize: number;
    customMetaData: Record<string, string>;
}
interface OnnxRuntimeWebOptions {
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
interface YoloOptions extends OnnxRuntimeWebOptions {
    /** ONNX model URL, ArrayBuffer, or Uint8Array. */
    model?: YoloModelSource;
    /** Browser execution provider priority. Defaults to ['wasm']. */
    executionProviders?: readonly YoloExecutionProvider[];
    /** Extra onnxruntime-web session options. */
    sessionOptions?: ort.InferenceSession.SessionOptions;
    /** Optional model type override for ONNX models without custom metadata. */
    modelType?: ModelType;
    /** Optional model version override for ONNX models without custom metadata. */
    modelVersion?: ModelVersion;
    /** Optional labels override for ONNX models without embedded class names. Accepts string[] or class_names.txt text. */
    labels?: YoloLabels;
    /** Resize mode used before inference. Defaults to proportional letterbox. */
    imageResize?: 'proportional' | 'stretch';
    /** Optional channel-wise input mean after scaling pixels to 0..1. */
    imageMean?: readonly [number, number, number];
    /** Optional channel-wise input standard deviation after scaling pixels to 0..1. */
    imageStd?: readonly [number, number, number];
    /** Optional preprocessing backend. Defaults to WebGPU when the WebGPU execution provider is used. */
    preprocessBackend?: 'cpu' | 'webgpu';
}
type YoloFeeds = ort.InferenceSession.FeedsType;
type YoloFetches = ort.InferenceSession.FetchesType;
type YoloRunOptions = ort.InferenceSession.RunOptions;
type YoloRunResult = ort.InferenceSession.ReturnType;
type YoloTensor = ort.Tensor;
interface IYoloHandler {
    preprocessImage(img: YoloImageSource, roi?: Rect | null): YoloPreprocessResult;
    RunObjectDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<ObjectDetection[]>;
    RunObbDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<OBBDetection[]>;
    RunSegmentation(img: YoloImageSource, confidence: number, pixelConfidence: number, iou: number, roi?: Rect | null): Promise<Segmentation[]>;
    RunPoseEstimation(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<PoseEstimation[]>;
    RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]>;
}

/**
 * Configure onnxruntime-web before creating an inference session.
 */
declare function initializeOnnxRuntimeWeb(options?: OnnxRuntimeWebOptions): void;

declare class Yolo {
    private readonly options;
    private readonly model?;
    private session;
    private _onnxModel;
    private _handler;
    private preprocessCanvas;
    private preprocessContext;
    private preprocessTensorData;
    private preprocessTensorSize;
    constructor(options?: YoloOptions);
    get yoloOptions(): YoloOptions;
    get preprocessBackend(): 'cpu' | 'webgpu';
    static create(options: YoloOptions): Promise<Yolo>;
    get isLoaded(): boolean;
    get inputNames(): readonly string[];
    get outputNames(): readonly string[];
    get onnxModel(): OnnxModel;
    load(model?: YoloModelSource): Promise<this>;
    run(feeds: YoloFeeds, options?: YoloRunOptions): Promise<YoloRunResult>;
    runWithFetches(feeds: YoloFeeds, fetches: YoloFetches, options?: YoloRunOptions): Promise<YoloRunResult>;
    predict(feeds: YoloFeeds, options?: YoloRunOptions): Promise<YoloRunResult>;
    RunObjectDetection(img: YoloImageSource, confidence?: number, iou?: number, roi?: Rect | null): Promise<ObjectDetection[]>;
    RunObbDetection(img: YoloImageSource, confidence?: number, iou?: number, roi?: Rect | null): Promise<OBBDetection[]>;
    RunSegmentation(img: YoloImageSource, confidence?: number, pixelConfidence?: number, iou?: number, roi?: Rect | null): Promise<Segmentation[]>;
    RunPoseEstimation(img: YoloImageSource, confidence?: number, iou?: number, roi?: Rect | null): Promise<PoseEstimation[]>;
    RunClassification(img: YoloImageSource, classes?: number): Promise<Classification[]>;
    preprocessImage(img: YoloImageSource, roi?: Rect | null): YoloPreprocessResult;
    scaleBoundingBox(x1: number, y1: number, x2: number, y2: number, input: YoloPreprocessResult): Rect;
    drawObjectDetections(source: YoloImageSource, detections: readonly ObjectDetection[], canvas: HTMLCanvasElement, options?: DetectionDrawingOptions): void;
    drawClassifications(source: YoloImageSource, classifications: readonly Classification[], canvas: HTMLCanvasElement, options?: ClassificationDrawingOptions): void;
    drawObbDetections(source: YoloImageSource, detections: readonly OBBDetection[], canvas: HTMLCanvasElement, options?: DetectionDrawingOptions): void;
    drawSegmentations(source: YoloImageSource, segmentations: readonly Segmentation[], canvas: HTMLCanvasElement, options?: SegmentationDrawingOptions): void;
    drawPoseEstimations(source: YoloImageSource, poseEstimations: readonly PoseEstimation[], canvas: HTMLCanvasElement, options?: PoseDrawingOptions): void;
    extractSegmentationEdgePoints(segmentation: Segmentation): {
        x: number;
        y: number;
    }[];
    extractSegmentationsEdgePoints(segmentations: readonly Segmentation[]): {
        x: number;
        y: number;
    }[][];
    tensor<T extends ort.Tensor.Type>(type: T, data: ort.Tensor.DataTypeMap[T], dims?: readonly number[]): ort.Tensor;
    getWebGpuDevice(): Promise<any>;
    tensorFromGpuBuffer(gpuBuffer: ort.Tensor.GpuBufferType, dims: readonly number[], dispose?: () => void): ort.Tensor;
    dispose(): Promise<void>;
    private createSessionOptions;
    private createSession;
    private ensureSession;
    private getInputShape;
    private getSourceRect;
    private calculateProportionalResize;
    private clamp;
    private getPreprocessContext;
    private getPreprocessTensorData;
    private getImageSourceSize;
    private ensureHandler;
    private requireModel;
    private hasWebGpuExecutionProvider;
    private isSupportedModel;
}

declare class DrawTool {
    static drawObjectDetections(source: YoloImageSource, detections: readonly ObjectDetection[], canvas: HTMLCanvasElement, options?: DetectionDrawingOptions): void;
    static drawClassifications(source: YoloImageSource, classifications: readonly Classification[], canvas: HTMLCanvasElement, options?: ClassificationDrawingOptions): void;
    static drawObbDetections(source: YoloImageSource, detections: readonly OBBDetection[], canvas: HTMLCanvasElement, options?: DetectionDrawingOptions): void;
    static drawSegmentations(source: YoloImageSource, segmentations: readonly Segmentation[], canvas: HTMLCanvasElement, options?: SegmentationDrawingOptions): void;
    static drawPoseEstimations(source: YoloImageSource, poseEstimations: readonly PoseEstimation[], canvas: HTMLCanvasElement, options?: PoseDrawingOptions): void;
    static extractSegmentationEdgePoints(segmentation: Segmentation): {
        x: number;
        y: number;
    }[];
    static extractSegmentationsEdgePoints(segmentations: readonly Segmentation[]): {
        x: number;
        y: number;
    }[][];
    private static traceOrderedEdgePoints;
    private static traceEdgeComponent;
    private static getNextEdgeNeighbor;
    private static getDirectionIndex;
    private static getTopLeftKey;
    private static drawBoundingBoxes;
    private static prepareDrawingCanvas;
    private static getDetectionColor;
    private static getDetectionDrawingAlpha;
    private static withAlpha;
    private static drawDetectionLabel;
    private static getCanvasFontSize;
    private static getObbCorners;
    private static drawSegmentationMask;
    private static drawSegmentationContour;
    static drawSegmentationEdgePoints(source: YoloImageSource, segmentations: readonly Segmentation[], canvas: HTMLCanvasElement, options?: SegmentationDrawingOptions): void;
    private static drawOrderedEdgePoints;
    private static isSegmentationEdgePixel;
    private static isPackedMaskSet;
    private static parseCanvasColor;
    private static drawPoseConnections;
    private static drawPoseConnection;
    private static getImageSourceSize;
    private static clamp;
}

export { Classification, type ClassificationDrawingOptions, type Detection, type DetectionDrawingOptions, DrawTool, type IYoloHandler, type KeyPoint, type KeyPointConnection, type KeyPointMarker, type LabelModel, type ModelDataType, type ModelType, type ModelVersion, OBBDetection, ObjectDetection, type OnnxModel, type OnnxRuntimeWebOptions, type Point, type PoseDrawingOptions, PoseEstimation, type Rect, Segmentation, type SegmentationDrawingOptions, TrackingInfo, Yolo, type YoloExecutionProvider, YoloExecutionProviderNames, YoloExecutionProviderOptions, type YoloFeeds, type YoloFetches, type YoloImageSource, type YoloLabels, type YoloModelSource, type YoloOptions, type YoloPreprocessResult, type YoloRunOptions, type YoloRunResult, type YoloTensor, YoloWebExecutionProviderOptions, initializeOnnxRuntimeWeb };
