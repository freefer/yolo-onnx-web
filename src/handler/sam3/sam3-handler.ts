import type * as OrtTypes from 'onnxruntime-web';
import { ort } from '../../runtime';
import type { Point, Rect, Segmentation, YoloImageSource } from '../../types';
import { ClipBpeTokenizer } from './clip-tokenizer';
import {
  bilinearResize,
  copyLowResMask,
  decodePcsOutputs,
  logitsToSegmentation,
} from './postprocess';
import {
  boxToModelCorners,
  clamp,
  pointToModel,
  pointToNormalized,
  preprocessSam3Image,
  rectToCxcywh,
  splitTextPrompts,
  composeTextQuery,
  packPcsGeometricPrompts,
} from './preprocess';
import { Sam3WebGpuPreprocessor, waitForWebGpuOutputs } from './webgpu-preprocess';
import type {
  Sam3InferenceState,
  Sam3MaskPrompt,
  Sam3PcsPrompt,
  Sam3PvsPrompt,
  Sam3PvsResult,
  Sam3VisionEmbeddings,
} from './types';

function asFloat32(data: OrtTypes.Tensor.DataType): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }

  return Float32Array.from(data as ArrayLike<number>);
}

function pickFeeds(
  session: OrtTypes.InferenceSession,
  candidates: Record<string, OrtTypes.Tensor | undefined>,
): OrtTypes.InferenceSession.FeedsType {
  const feeds: Record<string, OrtTypes.Tensor> = {};

  for (const name of session.inputNames) {
    const tensor = candidates[name];

    if (!tensor) {
      throw new Error(`SAM3 session is missing feed "${name}". Available: ${Object.keys(candidates).join(', ')}`);
    }

    feeds[name] = tensor;
  }

  return feeds;
}

function pickOutput(
  result: OrtTypes.InferenceSession.OnnxValueMapType,
  name: string,
): OrtTypes.Tensor {
  const tensor = result[name];

  if (!tensor) {
    throw new Error(`SAM3 output "${name}" is missing. Available: ${Object.keys(result).join(', ')}`);
  }

  return tensor;
}

function optionalOutput(
  result: OrtTypes.InferenceSession.OnnxValueMapType,
  name: string,
): OrtTypes.Tensor | undefined {
  return result[name];
}

function tensorFloat(data: Float32Array, dims: readonly number[]): OrtTypes.Tensor {
  return new ort.Tensor('float32', data, [...dims]);
}

function tensorInt64(values: ArrayLike<number>, dims: readonly number[]): OrtTypes.Tensor {
  const data = BigInt64Array.from(Array.from(values, value => BigInt(value)));
  return new ort.Tensor('int64', data, [...dims]);
}

function tensorInt32(values: ArrayLike<number>, dims: readonly number[]): OrtTypes.Tensor {
  return new ort.Tensor('int32', Int32Array.from(values), [...dims]);
}

function tensorBool(values: ArrayLike<boolean | number>, dims: readonly number[]): OrtTypes.Tensor {
  return new ort.Tensor('bool', Uint8Array.from(values, value => (value ? 1 : 0)), [...dims]);
}

function selectPvsMaskIndices(maskCount: number, ious: Float32Array, wantMulti: boolean): number[] {
  if (maskCount >= 4) {
    return wantMulti ? [1, 2, 3] : [0];
  }

  if (maskCount <= 1) {
    return [0];
  }

  return wantMulti ? Array.from({ length: maskCount }, (_, index) => index) : [argmax(ious)];
}

function prepareMaskInput(maskInput: Sam3MaskPrompt | null | undefined, maskSize: number): Float32Array {
  const maskData = new Float32Array(maskSize * maskSize);

  if (!maskInput) {
    return maskData;
  }

  if (maskInput.width === maskSize && maskInput.height === maskSize) {
    maskData.set(maskInput.logits.subarray(0, maskData.length));
    return maskData;
  }

  maskData.set(bilinearResize(maskInput.logits, maskInput.width, maskInput.height, maskSize, maskSize));
  return maskData;
}

export class Sam3Handler {
  private readonly webGpuPreprocessor: Sam3WebGpuPreprocessor | null;

  constructor(
    private readonly visionSession: OrtTypes.InferenceSession,
    private readonly textSession: OrtTypes.InferenceSession,
    private readonly groundingSession: OrtTypes.InferenceSession,
    private readonly promptSession: OrtTypes.InferenceSession | null,
    private readonly tokenizer: ClipBpeTokenizer,
    private readonly options: {
      confidenceThreshold: number;
      pixelConfidence: number;
      imageSize: number;
      maskSize: number;
      webGpu: boolean;
    },
  ) {
    this.webGpuPreprocessor = options.webGpu ? new Sam3WebGpuPreprocessor() : null;
  }

  setConfidenceThreshold(threshold: number, state?: Sam3InferenceState | null): number {
    this.options.confidenceThreshold = threshold;
    return state ? this.options.confidenceThreshold : threshold;
  }

  async applyConfidenceThreshold(state: Sam3InferenceState): Promise<Segmentation[]> {
    return this.forwardAllTextPrompts(state);
  }

  dispose(): void {
    this.webGpuPreprocessor?.dispose();
  }

  async setImage(image: YoloImageSource, state?: Sam3InferenceState | null): Promise<Sam3InferenceState> {
    if (state?.vision) {
      this.releaseVision(state.vision);
    }

    const preprocessStarted = performance.now();
    const input = await this.prepareImage(image);
    const preprocessMs = performance.now() - preprocessStarted;
    if (typeof (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler?.yield === 'function') {
      await (globalThis as { scheduler?: { yield: () => Promise<void> } }).scheduler!.yield();
    }
    let imageTensor = input.tensor ?? tensorFloat(input.data, [1, 3, input.height, input.width]);
    let result: OrtTypes.InferenceSession.OnnxValueMapType;
    let runMs = 0;
    let fenceMs = 0;

    try {
      try {
        const runStarted = performance.now();
        result = await this.visionSession.run(pickFeeds(this.visionSession, { images: imageTensor }));
        runMs = performance.now() - runStarted;
      } catch (error) {
        if (!input.tensor) {
          throw error;
        }

        console.warn('[SAM3] GPU image tensor was rejected. Falling back to CPU input.', error);
        imageTensor.dispose?.();
        const cpu = preprocessSam3Image(image, this.options.imageSize);
        imageTensor = tensorFloat(cpu.data, [1, 3, cpu.height, cpu.width]);
        const runStarted = performance.now();
        result = await this.visionSession.run(pickFeeds(this.visionSession, { images: imageTensor }));
        runMs = performance.now() - runStarted;
      }

      if (this.options.webGpu) {
        const fenceStarted = performance.now();
        await waitForWebGpuOutputs(result);
        fenceMs = performance.now() - fenceStarted;
      }
    } finally {
      imageTensor.dispose?.();
    }

    console.info(
      `[SAM3] encode preprocess=${preprocessMs.toFixed(0)}ms run=${runMs.toFixed(0)}ms fence=${fenceMs.toFixed(0)}ms total=${(preprocessMs + runMs + fenceMs).toFixed(0)}ms`,
    );

    return {
      sourceWidth: input.sourceWidth,
      sourceHeight: input.sourceHeight,
      vision: {
        detFpn0: pickOutput(result, 'det_fpn_0'),
        detFpn1: pickOutput(result, 'det_fpn_1'),
        detFpn2: pickOutput(result, 'det_fpn_2'),
        detPos0: optionalOutput(result, 'det_pos_0'),
        detPos1: optionalOutput(result, 'det_pos_1'),
        detPos2: optionalOutput(result, 'det_pos_2'),
        pvsHighRes0: pickOutput(result, 'pvs_high_res_0'),
        pvsHighRes1: pickOutput(result, 'pvs_high_res_1'),
        pvsImageEmbed: pickOutput(result, 'pvs_image_embed'),
      },
      boxes: [],
      points: [],
      pvsPoints: [],
      pvsBox: null,
      pvsMaskInput: null,
    };
  }

  resetPrompts(state: Sam3InferenceState): Sam3InferenceState {
    state.text = undefined;
    state.texts = undefined;
    state.textDescription = undefined;
    state.textEmbeddings = undefined;
    state.boxes = [];
    state.points = [];
    state.lastPcs = undefined;
    return state;
  }

  resetVisualPrompts(state: Sam3InferenceState): Sam3InferenceState {
    state.pvsPoints = [];
    state.pvsBox = null;
    state.pvsMaskInput = null;
    state.lastPvs = undefined;
    return state;
  }

  async setTextPrompt(prompt: string, state: Sam3InferenceState, description?: string): Promise<Segmentation[]> {
    this.ensureImage(state);
    const original = (prompt || 'visual').trim() || 'visual';
    state.text = original;
    state.textDescription = description?.trim() || undefined;
    state.texts = splitTextPrompts(original);
    if (state.texts.length === 0) {
      state.texts = [state.text];
    }
    return this.forwardAllTextPrompts(state);
  }

  async addGeometricPrompt(box: Rect, label: boolean, state: Sam3InferenceState): Promise<Segmentation[]> {
    this.ensureImage(state);
    state.boxes.push({ box, label });
    try {
      return await this.forwardAllTextPrompts(state);
    } catch (error) {
      state.boxes.pop();
      throw error;
    }
  }

  async addGeometricPoint(point: Point, label: boolean, state: Sam3InferenceState): Promise<Segmentation[]> {
    this.ensureImage(state);
    state.points.push({ point, label: label ? 1 : 0 });
    try {
      return await this.forwardAllTextPrompts(state);
    } catch (error) {
      state.points.pop();
      throw error;
    }
  }

  async removeGeometricPrompt(index: number, state: Sam3InferenceState): Promise<Segmentation[]> {
    if (index >= 0 && index < state.boxes.length) {
      state.boxes.splice(index, 1);
    }

    return this.forwardAllTextPrompts(state);
  }

  async removeGeometricPoint(index: number, state: Sam3InferenceState): Promise<Segmentation[]> {
    if (index >= 0 && index < state.points.length) {
      state.points.splice(index, 1);
    }

    return this.forwardAllTextPrompts(state);
  }

  async predictConcept(prompt: Sam3PcsPrompt, state: Sam3InferenceState): Promise<Segmentation[]> {
    if (prompt.text !== undefined) {
      const original = (prompt.text || 'visual').trim() || 'visual';
      state.text = original;
      state.texts = splitTextPrompts(original);
      if (state.texts.length === 0) {
        state.texts = [state.text];
      }
    }

    if (prompt.description !== undefined) {
      state.textDescription = prompt.description.trim() || undefined;
    }

    if (prompt.boxes) {
      state.boxes = [...prompt.boxes];
    }

    if (prompt.points) {
      state.points = [...prompt.points];
    }

    return this.forwardAllTextPrompts(state);
  }

  async predictVisual(prompt: Sam3PvsPrompt, state: Sam3InferenceState): Promise<Sam3PvsResult> {
    this.ensureImage(state);

    if (!this.promptSession) {
      throw new Error('SAM3 prompt decoder is not loaded. Pass promptDecoder when creating Sam3.');
    }

    const points = [...(prompt.points ?? state.pvsPoints)];
    const box = prompt.box === undefined ? state.pvsBox : prompt.box;
    const maskInput = prompt.maskInput === undefined ? state.pvsMaskInput : prompt.maskInput;
    const concatPoints: Array<{ x: number; y: number; label: number }> = [];

    if (box) {
      const corners = boxToModelCorners(box, state.sourceWidth, state.sourceHeight, this.options.imageSize);
      concatPoints.push({ x: corners[0][0], y: corners[0][1], label: 2 }, { x: corners[1][0], y: corners[1][1], label: 3 });
    }

    for (const item of points) {
      const [x, y] = pointToModel(item.point, state.sourceWidth, state.sourceHeight, this.options.imageSize);
      concatPoints.push({ x, y, label: item.label });
    }

    if (concatPoints.length === 0) {
      concatPoints.push({ x: 0, y: 0, label: -1 });
    }

    const coords = new Float32Array(concatPoints.length * 2);
    const labels = new Int32Array(concatPoints.length);

    for (let index = 0; index < concatPoints.length; index += 1) {
      coords[index * 2] = concatPoints[index].x;
      coords[index * 2 + 1] = concatPoints[index].y;
      labels[index] = concatPoints[index].label;
    }

    const maskSize = this.options.maskSize;
    const hasMask = maskInput ? 1 : 0;
    const result = await this.promptSession.run(
      pickFeeds(this.promptSession, {
        image_embed: state.vision.pvsImageEmbed,
        high_res_0: state.vision.pvsHighRes0,
        high_res_1: state.vision.pvsHighRes1,
        point_coords: tensorFloat(coords, [1, concatPoints.length, 2]),
        point_labels: tensorInt32(labels, [1, concatPoints.length]),
        mask_input: tensorFloat(prepareMaskInput(maskInput, maskSize), [1, 1, maskSize, maskSize]),
        has_mask_input: tensorFloat(Float32Array.from([hasMask]), [1]),
      }),
    );

    const lowRes = pickOutput(result, 'low_res_masks');
    const ious = asFloat32(pickOutput(result, 'iou_predictions').data);
    const objectScoreLogits = result.object_score_logits ? asFloat32(result.object_score_logits.data) : [];
    const objectScore = objectScoreLogits.length > 0 ? 1 / (1 + Math.exp(-(objectScoreLogits[0] ?? 0))) : 1;
    const lowResData = asFloat32(lowRes.data);
    const dims = lowRes.dims;
    const maskCount = dims.length >= 4 ? dims[1] : 1;
    const height = dims[dims.length - 2] ?? maskSize;
    const width = dims[dims.length - 1] ?? maskSize;
    const clickCount = concatPoints.filter(item => item.label >= 0 && item.label <= 1).length;
    const wantMulti = prompt.multimaskOutput ?? (clickCount <= 1 && !box && !maskInput);
    const selected = selectPvsMaskIndices(maskCount, ious, wantMulti);
    const masks: Segmentation[] = [];
    const lowResMasks: Float32Array[] = [];
    const selectedIous: number[] = [];
    const selectedScores: number[] = [];

    for (const index of selected) {
      const slice = copyLowResMask(lowResData, index, width, height);
      const predictedIou = ious[index] ?? 0;
      lowResMasks.push(slice);
      selectedIous.push(predictedIou);
      selectedScores.push(objectScore);
      masks.push(
        logitsToSegmentation(
          slice,
          width,
          height,
          state.sourceWidth,
          state.sourceHeight,
          'object',
          clamp(predictedIou, 0, 1),
          0,
          false,
        ),
      );
    }

    const pvsResult: Sam3PvsResult = {
      masks,
      lowResMasks,
      ious: selectedIous,
      objectScores: selectedScores,
    };
    state.pvsPoints = points;
    state.pvsBox = box;
    state.lastPvs = pvsResult;
    const bestIndex = argmax(selectedIous);
    state.pvsMaskInput = {
      logits: lowResMasks[bestIndex] ?? lowResMasks[0],
      width,
      height,
    };
    return pvsResult;
  }

  async addPoint(point: Point, label: 0 | 1, state: Sam3InferenceState): Promise<Sam3PvsResult> {
    state.pvsPoints.push({ point, label });
    const promptCount = state.pvsPoints.length + (state.pvsBox ? 1 : 0);
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box: state.pvsBox,
        maskInput: promptCount <= 1 ? null : state.pvsMaskInput,
        multimaskOutput: promptCount <= 1,
      },
      state,
    );
  }

  async addBox(box: Rect, state: Sam3InferenceState): Promise<Sam3PvsResult> {
    state.pvsBox = box;
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box,
        maskInput: state.pvsPoints.length === 0 ? null : state.pvsMaskInput,
        multimaskOutput: false,
      },
      state,
    );
  }

  async addMask(mask: Sam3MaskPrompt, state: Sam3InferenceState): Promise<Sam3PvsResult> {
    state.pvsMaskInput = mask;
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box: state.pvsBox,
        maskInput: mask,
        multimaskOutput: false,
      },
      state,
    );
  }

  async removePoint(index: number, state: Sam3InferenceState): Promise<Sam3PvsResult> {
    if (index >= 0 && index < state.pvsPoints.length) {
      state.pvsPoints.splice(index, 1);
    }

    state.pvsMaskInput = null;
    return this.predictVisual({ points: state.pvsPoints, box: state.pvsBox, maskInput: null }, state);
  }

  private async encodeText(prompt: string, state: Sam3InferenceState): Promise<void> {
    if (state.textEmbeddings) {
      this.releaseTensors(state.textEmbeddings);
      state.textEmbeddings = undefined;
    }

    const ids = this.tokenizer.tokenize(prompt);
    const result = await this.textSession.run(
      pickFeeds(this.textSession, {
        input_ids: tensorInt64(ids, [1, ids.length]),
      }),
    );
    state.textEmbeddings = {
      languageMask: pickOutput(result, 'language_mask'),
      languageFeatures: pickOutput(result, 'language_features'),
      languageEmbeds: pickOutput(result, 'language_embeds'),
    };
  }

  private async forwardAllTextPrompts(state: Sam3InferenceState): Promise<Segmentation[]> {
    this.ensureImage(state);
    const original = state.text?.trim() || 'visual';
    const texts = state.texts?.length ? state.texts : splitTextPrompts(original);
    const merged: Segmentation[] = [];

    for (const text of texts) {
      await this.encodeText(composeTextQuery(text, state.textDescription), state);
      merged.push(...(await this.forwardGrounding(state, text)));
    }

    state.text = original;
    state.texts = texts;
    state.lastPcs = merged;
    return merged;
  }

  private async forwardGrounding(state: Sam3InferenceState, label?: string): Promise<Segmentation[]> {
    this.ensureImage(state);

    if (!state.textEmbeddings) {
      throw new Error('Text embeddings are missing. Call setTextPrompt() first.');
    }

    const packed = packPcsGeometricPrompts(state.boxes, state.points);
    const boxes = packed.boxes;
    const points = packed.points;
    const boxCoords = new Float32Array(boxes.length * 4);
    const boxLabels = new Int32Array(boxes.length);
    const boxPadMask = new Uint8Array(boxes.length);

    for (let index = 0; index < boxes.length; index += 1) {
      const cxcywh = rectToCxcywh(boxes[index].box, state.sourceWidth, state.sourceHeight);
      boxCoords[index * 4] = cxcywh[0];
      boxCoords[index * 4 + 1] = cxcywh[1];
      boxCoords[index * 4 + 2] = cxcywh[2];
      boxCoords[index * 4 + 3] = cxcywh[3];
      boxLabels[index] = boxes[index].label ? 1 : 0;
      boxPadMask[index] = boxes[index].pad ? 1 : 0;
    }

    const pointCoords = new Float32Array(points.length * 2);
    const pointLabels = new Int32Array(points.length);
    const pointPadMask = new Uint8Array(points.length);

    for (let index = 0; index < points.length; index += 1) {
      const xy = pointToNormalized(points[index].point, state.sourceWidth, state.sourceHeight);
      pointCoords[index * 2] = xy[0];
      pointCoords[index * 2 + 1] = xy[1];
      pointLabels[index] = points[index].label > 0 ? 1 : 0;
      pointPadMask[index] = points[index].pad ? 1 : 0;
    }

    let result: OrtTypes.InferenceSession.OnnxValueMapType;
    try {
      result = await this.groundingSession.run(
        pickFeeds(this.groundingSession, {
          det_fpn_0: state.vision.detFpn0,
          det_fpn_1: state.vision.detFpn1,
          det_fpn_2: state.vision.detFpn2,
          det_pos_0: state.vision.detPos0,
          det_pos_1: state.vision.detPos1,
          det_pos_2: state.vision.detPos2,
          language_features: state.textEmbeddings.languageFeatures,
          language_mask: state.textEmbeddings.languageMask,
          box_coords: tensorFloat(boxCoords, [boxes.length, 1, 4]),
          box_labels: tensorInt64(boxLabels, [boxes.length, 1]),
          box_pad_mask: tensorBool(boxPadMask, [1, boxes.length]),
          point_coords: tensorFloat(pointCoords, [points.length, 1, 2]),
          point_labels: tensorInt64(pointLabels, [points.length, 1]),
          point_pad_mask: tensorBool(pointPadMask, [1, points.length]),
        }),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `SAM3 grounding decoder failed with ${state.boxes.length} box(es) and ${state.points.length} point(s): ${message}`,
      );
    }

    const masks = decodePcsOutputs(
      asFloat32(pickOutput(result, 'pred_masks').data),
      asFloat32(pickOutput(result, 'pred_boxes').data),
      asFloat32(pickOutput(result, 'pred_logits').data),
      asFloat32(pickOutput(result, 'presence_logits').data),
      pickOutput(result, 'pred_masks').dims,
      pickOutput(result, 'pred_boxes').dims,
      state.sourceWidth,
      state.sourceHeight,
      label ?? state.texts?.[0] ?? state.text ?? 'visual',
      this.options.confidenceThreshold,
      this.options.pixelConfidence,
    );
    state.lastPcs = masks;
    return masks;
  }

  private async prepareImage(image: YoloImageSource) {
    if (this.webGpuPreprocessor) {
      try {
        return await this.webGpuPreprocessor.process(image, this.options.imageSize);
      } catch (error) {
        console.warn('[SAM3] WebGPU preprocessing failed. Falling back to CPU preprocessing.', error);
      }
    }

    return preprocessSam3Image(image, this.options.imageSize);
  }

  private ensureImage(state: Sam3InferenceState): void {
    if (!state?.vision) {
      throw new Error('You must call setImage() before prompting SAM3.');
    }
  }

  private releaseTensors(tensors: object): void {
    for (const tensor of Object.values(tensors as Record<string, { dispose?: () => void } | undefined>)) {
      tensor?.dispose?.();
    }
  }

  private releaseVision(vision: Sam3VisionEmbeddings): void {
    this.releaseTensors(vision);
  }
}

function argmax(values: ArrayLike<number>): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index += 1) {
    if ((values[index] ?? Number.NEGATIVE_INFINITY) > bestValue) {
      bestValue = values[index] ?? Number.NEGATIVE_INFINITY;
      bestIndex = index;
    }
  }

  return bestIndex;
}
