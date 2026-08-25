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
}

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
  languageEmbeds: ort.Tensor;
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
  lastPvs?: Sam3PvsResult;
}

export interface Sam3PvsResult {
  masks: Segmentation[];
  lowResMasks: Float32Array[];
  ious: number[];
  objectScores: number[];
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
}

export type Sam3ImageInput = YoloImageSource;
