import type { Point, Rect, Segmentation, YoloImageSource, YoloModelSource } from '../../types';
import type { OnnxRuntimeWebOptions, YoloExecutionProvider } from '../../types';
import type * as ort from 'onnxruntime-web';

export type Sam3ModelSource = YoloModelSource;
export type Sam3PromptLabel = -1 | 0 | 1 | 2 | 3;

export interface Sam3BoxPrompt {
  box: Rect;
  /** true = 正例范例，false = 负例范例 */
  label: boolean;
}

export interface Sam3PointPrompt {
  point: Point;
  /** 1 = 前景，0 = 背景，-1 = padding，2/3 = 框角点 */
  label: Sam3PromptLabel;
}

export interface Sam3MaskPrompt {
  /** 上一轮 PVS 的低分辨率 logits，通常为 288x288。 */
  logits: Float32Array;
  width: number;
  height: number;
}

export interface Sam3PcsPrompt {
  text?: string;
  description?: string;
  boxes?: readonly Sam3BoxPrompt[];
  points?: readonly Sam3PointPrompt[];
}

export interface Sam3PvsPrompt {
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

export interface Sam3PvsResult {
  masks: Segmentation[];
  lowResMasks: Float32Array[];
  ious: number[];
  objectScores: number[];
  maskWidth?: number;
  maskHeight?: number;
}

export type Sam3HoverMode = 'replace' | 'accumulate';
export type Sam3HoverPick = 'smallest' | 'iou' | 'first';

export interface Sam3HoverSelectOptions {
  /** 多候选选择：含提示点时默认 smallest，否则 iou */
  pick?: Sam3HoverPick;
  /** 只保留提示点附近的连通域，默认 true */
  isolateComponent?: boolean;
  promptPoint?: Point;
}

export interface Sam3HoverPointOptions extends Sam3HoverSelectOptions {
  label?: 0 | 1;
  /** replace：每次独立单点（默认，适合 mousemove）；accumulate：等同 addPoint */
  mode?: Sam3HoverMode;
  multimaskOutput?: boolean;
  /** 使用上一轮 low-res mask 作为 mask_input 做精细化，replace 默认 false */
  refinePrevious?: boolean;
}

export interface Sam3HoverBoxOptions extends Sam3HoverSelectOptions {
  multimaskOutput?: boolean;
  refinePrevious?: boolean;
}

export interface Sam3HoverResult extends Sam3PvsResult {
  mask: Segmentation | null;
  /** `mask` 对应的原始候选下标，便于把 low-res logits 写回 PVS 状态 */
  maskIndex: number;
  promptPoint?: Point;
  promptBox?: Rect;
}

export interface Sam3ImagePixelMask {
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

export interface Sam3MaskRasterOptions {
  imageWidth: number;
  imageHeight: number;
  sourceWidth?: number;
  sourceHeight?: number;
}

export interface Sam3MaskOverlayOptions {
  fill?: string;
  stroke?: string;
  dest?: Rect;
  displayWidth?: number;
  displayHeight?: number;
  sourceWidth?: number;
  sourceHeight?: number;
}

export interface Sam3MaskPolygonOptions {
  imageWidth: number;
  imageHeight: number;
  sourceWidth?: number;
  sourceHeight?: number;
  prompt?: Point;
  epsilon?: number;
  /** 单条轮廓最多保留的点数，默认 96 */
  maxPoints?: number;
}

export interface Sam3PointerToImageOptions {
  imageWidth?: number;
  imageHeight?: number;
  /** client：用 getBoundingClientRect 映射（默认）；offset：用 offsetX/offsetY */
  origin?: 'client' | 'offset';
  objectFit?: 'fill' | 'contain' | 'cover' | 'none';
}

export interface Sam3HoverPreviewOptions extends Sam3HoverPointOptions {
  /** 与上一次推理点的最小移动距离，默认 2 */
  minMove?: number;
  /** 点击确认时，与最近一次悬停点的复用距离（编码图像像素），默认 8 */
  confirmMaxDistance?: number;
  onResult?: (result: Sam3HoverResult | null) => void;
}

export type Sam3PointerLike = {
  clientX: number;
  clientY: number;
  offsetX?: number;
  offsetY?: number;
};

export interface Sam3VisionEmbeddings {
  detFpn0: ort.Tensor;
  detFpn1: ort.Tensor;
  detFpn2: ort.Tensor;
  detPos0?: ort.Tensor;
  detPos1?: ort.Tensor;
  detPos2?: ort.Tensor;
  pvsHighRes0: ort.Tensor;
  pvsHighRes1: ort.Tensor;
  pvsImageEmbed: ort.Tensor;
}

export interface Sam3TextEmbeddings {
  languageMask: ort.Tensor;
  languageFeatures: ort.Tensor;
  languageEmbeds?: ort.Tensor;
}

export interface Sam3PcsRawOutput {
  predMasks: Float32Array;
  predBoxes: Float32Array;
  predLogits: Float32Array;
  presenceLogits: Float32Array;
  maskShape: number[];
  boxShape: number[];
  label: string;
}

export interface Sam3InferenceState {
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

export interface Sam3Options extends OnnxRuntimeWebOptions {
  visionEncoder: Sam3ModelSource;
  textEncoder: Sam3ModelSource;
  groundingDecoder: Sam3ModelSource;
  promptDecoder?: Sam3ModelSource;
  tokenizer?: string | Sam3TokenizerTables;
  executionProviders?: readonly YoloExecutionProvider[];
  sessionOptions?: ort.InferenceSession.SessionOptions;
  confidenceThreshold?: number;
  pixelConfidence?: number;
  imageSize?: number;
  maskSize?: number;
  textLength?: number;
  onLoadProgress?: (message: string) => void;
}

export interface Sam3TokenizerTables {
  context_length: number;
  sot_token_id: number;
  eot_token_id: number;
  encoder: Record<string, number>;
  bpe_ranks: Record<string, number>;
  byte_encoder: Record<string, string>;
}

export interface Sam3ImageTensor {
  data: Float32Array;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  /** WebGPU 预处理得到的 GPU 输入，存在时编码不再走 CPU Float32Array。 */
  tensor?: ort.Tensor;
}

export type Sam3ImageInput = YoloImageSource;
