import * as onnxruntime_web_webgpu from 'onnxruntime-web/webgpu';
import * as ort$1 from 'onnxruntime-web';

type YoloModelSource = string | ArrayBufferLike | Uint8Array;
type YoloImageSource = CanvasImageSource;
type YoloExecutionProvider = NonNullable<ort$1.InferenceSession.SessionOptions['executionProviders']>[number];
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
    pixelMaskWidth?: number;
    pixelMaskHeight?: number;
    constructor(options: Detection & {
        bitPackedPixelMask: Uint8Array;
        segmentationEdgePoints?: Point[];
        pixelMaskWidth?: number;
        pixelMaskHeight?: number;
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
    /**
     * Which onnxruntime-web package entry to load.
     * Defaults to `auto`:
     * - webgpu → `onnxruntime-web/jspi` when JSPI is available, otherwise `onnxruntime-web/webgpu`
     * - webgl → `onnxruntime-web/webgl`
     * - webnn → `onnxruntime-web/all`
     * - otherwise → `onnxruntime-web/wasm`
     */
    ortBundle?: 'auto' | 'webgpu' | 'jspi' | 'wasm' | 'webgl' | 'all';
}
interface YoloOptions extends OnnxRuntimeWebOptions {
    /** ONNX model URL, ArrayBuffer, or Uint8Array. */
    model?: YoloModelSource;
    /** Browser execution provider priority. Defaults to ['wasm']. */
    executionProviders?: readonly YoloExecutionProvider[];
    /** Extra onnxruntime-web session options. */
    sessionOptions?: ort$1.InferenceSession.SessionOptions;
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
type YoloFeeds = ort$1.InferenceSession.FeedsType;
type YoloFetches = ort$1.InferenceSession.FetchesType;
type YoloRunOptions = ort$1.InferenceSession.RunOptions;
type YoloRunResult = ort$1.InferenceSession.ReturnType;
type YoloTensor = ort$1.Tensor;
interface IYoloHandler {
    preprocessImage(img: YoloImageSource, roi?: Rect | null): YoloPreprocessResult;
    RunObjectDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<ObjectDetection[]>;
    RunObbDetection(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<OBBDetection[]>;
    RunSegmentation(img: YoloImageSource, confidence: number, pixelConfidence: number, iou: number, roi?: Rect | null): Promise<Segmentation[]>;
    RunPoseEstimation(img: YoloImageSource, confidence: number, iou: number, roi?: Rect | null): Promise<PoseEstimation[]>;
    RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]>;
}

type OrtBundle = 'auto' | 'webgpu' | 'jspi' | 'wasm' | 'webgl' | 'all';
type OrtModule = typeof onnxruntime_web_webgpu;
/**
 * Chrome 137+ / Edge expose JSPI as `WebAssembly.Suspending`.
 * The JSPI WebGPU build avoids Asyncify, which otherwise blocks the UI during SAM3 encode.
 */
declare function isWebAssemblyJspiAvailable(): boolean;
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
declare function resolveOrtBundle(executionProviders?: readonly YoloExecutionProvider[], ortBundle?: OrtBundle): Exclude<OrtBundle, 'auto'>;
/** Whether an already-loaded ORT entry can serve the requested entry without a page reload. */
declare function canReuseOrtBundle(loaded: Exclude<OrtBundle, 'auto'>, requested: Exclude<OrtBundle, 'auto'>): boolean;
/**
 * Configure onnxruntime-web before creating an inference session.
 * Lazily loads the matching npm package entry from `executionProviders` / `ortBundle`.
 */
declare function initializeOnnxRuntimeWeb(options?: YoloOptions): Promise<OrtModule>;
declare function ensureOnnxRuntimeWebInitialized(options?: YoloOptions): Promise<OrtModule>;
declare function getOrt(): OrtModule;
declare function getLoadedOrtBundle(): Exclude<OrtBundle, 'auto'> | null;
/** Compatible alias used after initialization (Tensor / InferenceSession / env). */
declare const ort: OrtModule;

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
    tensor<T extends ort$1.Tensor.Type>(type: T, data: ort$1.Tensor.DataTypeMap[T], dims?: readonly number[]): ort$1.Tensor;
    getWebGpuDevice(): Promise<any>;
    tensorFromGpuBuffer(gpuBuffer: ort$1.Tensor.GpuBufferType, dims: readonly number[], dispose?: () => void): ort$1.Tensor;
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

type Sam3ModelSource = YoloModelSource;
type Sam3PromptLabel = -1 | 0 | 1 | 2 | 3;
interface Sam3BoxPrompt {
    box: Rect;
    /** true = 正例范例，false = 负例范例 */
    label: boolean;
}
interface Sam3PointPrompt {
    point: Point;
    /** 1 = 前景，0 = 背景，-1 = padding，2/3 = 框角点 */
    label: Sam3PromptLabel;
}
interface Sam3MaskPrompt {
    /** 上一轮 PVS 的低分辨率 logits，通常为 288x288。 */
    logits: Float32Array;
    width: number;
    height: number;
}
interface Sam3PcsPrompt {
    text?: string;
    description?: string;
    boxes?: readonly Sam3BoxPrompt[];
    points?: readonly Sam3PointPrompt[];
}
interface Sam3PvsPrompt {
    points?: readonly Sam3PointPrompt[];
    box?: Rect | null;
    maskInput?: Sam3MaskPrompt | null;
    multimaskOutput?: boolean;
    /**
     * 是否把本次提示写入 inferenceState（点/框/mask_input/lastPvs）。
     * 悬停预览应设为 false，避免冲掉已累积的 PVS 会话。
     */
    persist?: boolean;
}
interface Sam3PvsResult {
    masks: Segmentation[];
    lowResMasks: Float32Array[];
    ious: number[];
    objectScores: number[];
    maskWidth?: number;
    maskHeight?: number;
}
type Sam3HoverMode = 'replace' | 'accumulate';
type Sam3HoverPick = 'smallest' | 'iou' | 'first';
interface Sam3HoverSelectOptions {
    /** 多候选选择：含提示点时默认 smallest，否则 iou */
    pick?: Sam3HoverPick;
    /** 只保留提示点附近的连通域，默认 true */
    isolateComponent?: boolean;
    promptPoint?: Point;
}
interface Sam3HoverPointOptions extends Sam3HoverSelectOptions {
    label?: 0 | 1;
    /** replace：每次独立单点（默认，适合 mousemove）；accumulate：等同 addPoint */
    mode?: Sam3HoverMode;
    multimaskOutput?: boolean;
    /** 使用上一轮 low-res mask 作为 mask_input 做精细化，replace 默认 false */
    refinePrevious?: boolean;
}
interface Sam3HoverBoxOptions extends Sam3HoverSelectOptions {
    multimaskOutput?: boolean;
    refinePrevious?: boolean;
}
interface Sam3HoverResult extends Sam3PvsResult {
    mask: Segmentation | null;
    /** `mask` 对应的原始候选下标，便于把 low-res logits 写回 PVS 状态 */
    maskIndex: number;
    promptPoint?: Point;
    promptBox?: Rect;
}
interface Sam3ImagePixelMask {
    x: number;
    y: number;
    width: number;
    height: number;
    area: number;
    /** 图像坐标系下 0/1 像素 */
    pixels: Uint8Array;
    /** 与 pixels 对应的 bit-packed 掩码 */
    packed: Uint8Array;
}
interface Sam3MaskRasterOptions {
    imageWidth: number;
    imageHeight: number;
    sourceWidth?: number;
    sourceHeight?: number;
}
interface Sam3MaskOverlayOptions {
    fill?: string;
    stroke?: string;
    dest?: Rect;
    displayWidth?: number;
    displayHeight?: number;
    sourceWidth?: number;
    sourceHeight?: number;
}
interface Sam3MaskPolygonOptions {
    imageWidth: number;
    imageHeight: number;
    sourceWidth?: number;
    sourceHeight?: number;
    prompt?: Point;
    epsilon?: number;
    /** 单条轮廓最多保留的点数，默认 96 */
    maxPoints?: number;
}
interface Sam3PointerToImageOptions {
    imageWidth?: number;
    imageHeight?: number;
    /** client：用 getBoundingClientRect 映射（默认）；offset：用 offsetX/offsetY */
    origin?: 'client' | 'offset';
    objectFit?: 'fill' | 'contain' | 'cover' | 'none';
}
interface Sam3HoverPreviewOptions extends Sam3HoverPointOptions {
    /** 与上一次推理点的最小移动距离，默认 2 */
    minMove?: number;
    /** 点击确认时，与最近一次悬停点的复用距离（编码图像像素），默认 8 */
    confirmMaxDistance?: number;
    onResult?: (result: Sam3HoverResult | null) => void;
}
type Sam3PointerLike = {
    clientX: number;
    clientY: number;
    offsetX?: number;
    offsetY?: number;
};
interface Sam3VisionEmbeddings {
    detFpn0: ort$1.Tensor;
    detFpn1: ort$1.Tensor;
    detFpn2: ort$1.Tensor;
    detPos0?: ort$1.Tensor;
    detPos1?: ort$1.Tensor;
    detPos2?: ort$1.Tensor;
    pvsHighRes0: ort$1.Tensor;
    pvsHighRes1: ort$1.Tensor;
    pvsImageEmbed: ort$1.Tensor;
}
interface Sam3TextEmbeddings {
    languageMask: ort$1.Tensor;
    languageFeatures: ort$1.Tensor;
    languageEmbeds?: ort$1.Tensor;
}
interface Sam3PcsRawOutput {
    predMasks: Float32Array;
    predBoxes: Float32Array;
    predLogits: Float32Array;
    presenceLogits: Float32Array;
    maskShape: number[];
    boxShape: number[];
    label: string;
}
interface Sam3InferenceState {
    sourceWidth: number;
    sourceHeight: number;
    vision: Sam3VisionEmbeddings;
    text?: string;
    texts?: string[];
    textDescription?: string;
    textEmbeddings?: Sam3TextEmbeddings;
    boxes: Sam3BoxPrompt[];
    points: Sam3PointPrompt[];
    pvsPoints: Sam3PointPrompt[];
    pvsBox: Rect | null;
    pvsMaskInput: Sam3MaskPrompt | null;
    lastPcs?: Segmentation[];
    lastPcsRaw?: Sam3PcsRawOutput[];
    lastPvs?: Sam3PvsResult;
}
interface Sam3Options extends OnnxRuntimeWebOptions {
    visionEncoder: Sam3ModelSource;
    textEncoder: Sam3ModelSource;
    groundingDecoder: Sam3ModelSource;
    promptDecoder?: Sam3ModelSource;
    tokenizer?: string | Sam3TokenizerTables;
    executionProviders?: readonly YoloExecutionProvider[];
    sessionOptions?: ort$1.InferenceSession.SessionOptions;
    confidenceThreshold?: number;
    pixelConfidence?: number;
    imageSize?: number;
    maskSize?: number;
    textLength?: number;
    onLoadProgress?: (message: string) => void;
}
interface Sam3TokenizerTables {
    context_length: number;
    sot_token_id: number;
    eot_token_id: number;
    encoder: Record<string, number>;
    bpe_ranks: Record<string, number>;
    byte_encoder: Record<string, string>;
}
type Sam3ImageInput = YoloImageSource;

declare const SAM3_IMAGE_SIZE = 1008;
declare const SAM3_MASK_SIZE = 288;
declare const SAM3_TEXT_LENGTH = 32;
declare function splitTextPrompts(prompt: string): string[];

interface Sam3HoverHost {
    hoverPoint(point: Point, options?: Sam3HoverPointOptions): Promise<Sam3HoverResult>;
    hoverFromPointer(event: Sam3PointerLike, target: HTMLElement | DOMRect, options?: Sam3HoverPointOptions & Sam3PointerToImageOptions): Promise<Sam3HoverResult>;
    hoverFromDisplayPoint(point: Point, displayWidth: number, displayHeight: number, options?: Sam3HoverPointOptions): Promise<Sam3HoverResult>;
}
/**
 * 鼠标悬停预览：只保留最新一点，推理中的中间点丢弃。
 */
declare class Sam3HoverPreview {
    private readonly host;
    private readonly options;
    private queued;
    private queuedKind;
    private queuedDisplay;
    private running;
    private lastPoint;
    private lastResult;
    onResult: ((result: Sam3HoverResult | null) => void) | null;
    constructor(host: Sam3HoverHost, options?: Sam3HoverPreviewOptions);
    get result(): Sam3HoverResult | null;
    get mask(): Segmentation | null;
    get busy(): boolean;
    idle(): Promise<void>;
    /** 已编码图像坐标系中的点 */
    queuePoint(point: Point): void;
    /** 显示坐标系中的点（例如标注画布上的图像像素） */
    queueDisplayPoint(point: Point, displayWidth: number, displayHeight: number): void;
    /** Pointer / Mouse 事件，自动映射到图像像素再编码 */
    queuePointer(event: Sam3PointerLike, target: HTMLElement | DOMRect, pointer?: Sam3PointerToImageOptions): void;
    /** offsetX / offsetY（元素 CSS 像素） */
    queueOffset(offsetX: number, offsetY: number, target: HTMLElement, pointer?: Sam3PointerToImageOptions): void;
    clear(): void;
    /** 编码图坐标是否足够接近最近一次悬停结果，可直接作为点击确认。 */
    canReusePoint(point: Point, maxDistance?: number): boolean;
    /**
     * 点击确认：距离最近悬停点足够近时直接返回预览掩码，否则按同一套 hover 后处理再推理。
     */
    confirmPoint(point: Point, maxDistance?: number): Promise<Sam3HoverResult>;
    private kick;
    private flush;
}

declare function maskToPolygons(mask: Segmentation, options: Sam3MaskPolygonOptions): number[][][];
declare function maskToPolygon(mask: Segmentation, options: Sam3MaskPolygonOptions): number[][];

declare function pickBestMask(masks: readonly Segmentation[], ious?: readonly number[], options?: {
    pick?: Sam3HoverPick;
    point?: Point;
}): Segmentation | null;
/**
 * 从 PVS 多候选中选出预览/点击应展示的那一块：默认点选 smallest + 连通域裁剪。
 */
declare function selectVisualMask(result: Sam3PvsResult, options?: Sam3HoverSelectOptions, promptBox?: Rect | null): Sam3HoverResult;
declare function isolateMaskComponent(mask: Segmentation, point: Point): Segmentation;
/** 把编码图上的 Segmentation 采样到显示/标注图像坐标系。 */
declare function maskToImagePixels(mask: Segmentation, options: Sam3MaskRasterOptions): Sam3ImagePixelMask | null;

declare function pointerToImagePoint(event: Sam3PointerLike, target: HTMLElement | DOMRect, options?: Sam3PointerToImageOptions): Point;
declare function mapImagePoint(point: Point, from: {
    width: number;
    height: number;
}, to: {
    width: number;
    height: number;
}): Point;
declare function mapImageBox(box: Rect, from: {
    width: number;
    height: number;
}, to: {
    width: number;
    height: number;
}): Rect;

declare class Sam3 {
    private readonly options;
    private visionSession;
    private textSession;
    private groundingSession;
    private promptSession;
    private handler;
    private tokenizer;
    private state;
    private readonly webGpu;
    constructor(options: Sam3Options);
    static create(options: Sam3Options): Promise<Sam3>;
    get isLoaded(): boolean;
    get inferenceState(): Sam3InferenceState | null;
    load(): Promise<this>;
    setImage(image: Sam3ImageInput): Promise<Sam3InferenceState>;
    setTextPrompt(prompt: string, description?: string): Promise<Segmentation[]>;
    addGeometricPrompt(box: Rect, label?: boolean): Promise<Segmentation[]>;
    addGeometricPoint(point: Point, label?: boolean): Promise<Segmentation[]>;
    removeGeometricPrompt(index: number): Promise<Segmentation[]>;
    removeGeometricPoint(index: number): Promise<Segmentation[]>;
    resetPrompts(): Sam3InferenceState;
    resetVisualPrompts(): Sam3InferenceState;
    setConfidenceThreshold(threshold: number): Promise<Segmentation[] | Sam3InferenceState>;
    predictConcept(prompt: Sam3PcsPrompt): Promise<Segmentation[]>;
    predictVisual(prompt?: Sam3PvsPrompt): Promise<Sam3PvsResult>;
    addPoint(point: Point, label?: 0 | 1): Promise<Sam3PvsResult>;
    addBox(box: Rect): Promise<Sam3PvsResult>;
    addMask(mask: Sam3MaskPrompt): Promise<Sam3PvsResult>;
    /**
     * 悬停/点选预览：默认每次独立单点，不写入 PVS 累积状态。
     * 填充请用 {@link drawMask}（像素掩码），不要把拼接边点当一个多边形描边。
     */
    hoverPoint(point: Point, options?: Sam3HoverPointOptions): Promise<Sam3HoverResult>;
    hoverBox(box: Rect, options?: Sam3HoverBoxOptions): Promise<Sam3HoverResult>;
    hoverPoints(points: readonly Sam3PointPrompt[], options?: Sam3HoverSelectOptions & {
        refinePrevious?: boolean;
        multimaskOutput?: boolean;
    }): Promise<Sam3HoverResult>;
    hoverVisual(prompt: Sam3PvsPrompt, options?: Sam3HoverSelectOptions, promptBox?: Rect | null): Promise<Sam3HoverResult>;
    /** PointerEvent / MouseEvent → 编码图坐标后悬停 */
    hoverFromPointer(event: Sam3PointerLike, target: HTMLElement | DOMRect, options?: Sam3HoverPointOptions & Sam3PointerToImageOptions): Promise<Sam3HoverResult>;
    /** 显示画布坐标 → 编码图坐标后悬停 */
    hoverFromDisplayPoint(point: Point, displayWidth: number, displayHeight: number, options?: Sam3HoverPointOptions): Promise<Sam3HoverResult>;
    /** offsetX / offsetY（CSS 像素）→ 编码图坐标后悬停 */
    hoverFromOffset(offsetX: number, offsetY: number, target: HTMLElement, options?: Sam3HoverPointOptions & Sam3PointerToImageOptions): Promise<Sam3HoverResult>;
    createHoverPreview(options?: Sam3HoverPreviewOptions): Sam3HoverPreview;
    toEncodedPoint(point: Point, displayWidth: number, displayHeight: number): Point;
    toEncodedBox(box: Rect, displayWidth: number, displayHeight: number): Rect;
    toDisplayPoint(point: Point, displayWidth: number, displayHeight: number): Point;
    static pointerToImagePoint(event: Sam3PointerLike, target: HTMLElement | DOMRect, options?: Sam3PointerToImageOptions): Point;
    static mapPoint(point: Point, from: {
        width: number;
        height: number;
    }, to: {
        width: number;
        height: number;
    }): Point;
    /**
     * 在已有 canvas 上下文上绘制像素掩码。边界用掩码边缘像素着色，
     * 不要把 `segmentationEdgePoints` 当单个闭合折线 fill/stroke。
     */
    drawMask(context: CanvasRenderingContext2D, mask: Segmentation, options?: Sam3MaskOverlayOptions): void;
    maskToPolygon(mask: Segmentation, options: Sam3MaskPolygonOptions): number[][];
    maskToPolygons(mask: Segmentation, options: Sam3MaskPolygonOptions): number[][][];
    toImagePixelMask(mask: Segmentation, imageWidth: number, imageHeight: number, options?: Omit<Sam3MaskRasterOptions, 'imageWidth' | 'imageHeight'>): Sam3ImagePixelMask | null;
    selectVisualMask(result: Sam3PvsResult, options?: Sam3HoverSelectOptions, promptBox?: Rect | null): Sam3HoverResult;
    /** 把 hover / confirm 选中的候选写入 PVS 状态，不再次推理。 */
    acceptVisualResult(result: Sam3HoverResult): Sam3HoverResult;
    selectVisualCandidate(index: number): Sam3PvsResult;
    removePoint(index: number): Promise<Sam3PvsResult>;
    drawSegmentations(source: Sam3ImageInput, segmentations: readonly Segmentation[], canvas: HTMLCanvasElement, options?: SegmentationDrawingOptions): void;
    drawSegmentationEdgePoints(source: Sam3ImageInput, segmentations: readonly Segmentation[], canvas: HTMLCanvasElement, options?: SegmentationDrawingOptions): void;
    dispose(): Promise<void>;
    private createSessionOptions;
    private createExecutionProviders;
    private createSession;
    private createSessionWithOptions;
    private encodedSize;
    private pointerImageSize;
    private selectHoverMask;
    private withSourceSize;
    private ensureHandler;
    private ensureState;
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
    static extractSegmentationContours(segmentation: Segmentation): Point[][];
    static splitEdgeContours(points: readonly Point[]): Point[][];
    static estimateEdgeStep(points: readonly Point[]): number;
    static drawPackedMaskOverlay(context: CanvasRenderingContext2D, segmentation: Segmentation, options?: {
        fill?: string;
        stroke?: string;
        dest?: Rect;
    }): void;
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
    private static drawOrderedEdgeContours;
    private static isMostlyClosedContour;
    private static resolvePackedMaskSize;
    private static isSegmentationEdgePixel;
    private static isPackedMaskSet;
    private static parseCanvasColor;
    private static drawPoseConnections;
    private static drawPoseConnection;
    private static getImageSourceSize;
    private static clamp;
}

export { Classification, type ClassificationDrawingOptions, type Detection, type DetectionDrawingOptions, DrawTool, type IYoloHandler, type KeyPoint, type KeyPointConnection, type KeyPointMarker, type LabelModel, type ModelDataType, type ModelType, type ModelVersion, OBBDetection, ObjectDetection, type OnnxModel, type OnnxRuntimeWebOptions, type OrtBundle, type OrtModule, type Point, type PoseDrawingOptions, PoseEstimation, type Rect, SAM3_IMAGE_SIZE, SAM3_MASK_SIZE, SAM3_TEXT_LENGTH, Sam3, type Sam3BoxPrompt, type Sam3HoverBoxOptions, type Sam3HoverMode, type Sam3HoverPick, type Sam3HoverPointOptions, Sam3HoverPreview, type Sam3HoverPreviewOptions, type Sam3HoverResult, type Sam3HoverSelectOptions, type Sam3ImageInput, type Sam3ImagePixelMask, type Sam3InferenceState, type Sam3MaskOverlayOptions, type Sam3MaskPolygonOptions, type Sam3MaskPrompt, type Sam3MaskRasterOptions, type Sam3Options, type Sam3PcsPrompt, type Sam3PcsRawOutput, type Sam3PointPrompt, type Sam3PointerLike, type Sam3PointerToImageOptions, type Sam3PvsPrompt, type Sam3PvsResult, type Sam3TokenizerTables, Segmentation, type SegmentationDrawingOptions, TrackingInfo, Yolo, type YoloExecutionProvider, YoloExecutionProviderNames, YoloExecutionProviderOptions, type YoloFeeds, type YoloFetches, type YoloImageSource, type YoloLabels, type YoloModelSource, type YoloOptions, type YoloPreprocessResult, type YoloRunOptions, type YoloRunResult, type YoloTensor, YoloWebExecutionProviderOptions, canReuseOrtBundle, ensureOnnxRuntimeWebInitialized, getLoadedOrtBundle, getOrt, initializeOnnxRuntimeWeb, isWebAssemblyJspiAvailable, isolateMaskComponent, mapImageBox, mapImagePoint, maskToImagePixels, maskToPolygon, maskToPolygons, ort, pickBestMask, pointerToImagePoint, resolveOrtBundle, selectVisualMask, splitTextPrompts };
