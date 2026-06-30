import {
  Classification,
  IYoloHandler,
  OBBDetection,
  ObjectDetection,
  PoseEstimation,
  Rect,
  Segmentation,
  YoloImageSource,
  YoloPreprocessResult,
} from '../../types';
import { Yolo } from '../../yolo';
import { clamp, unsupportedTask } from '../common';

const BACKGROUND_CLASS_PREFIX = 'background_class';
const DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406] as const;
const DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225] as const;

export class RF_DETRHandler implements IYoloHandler {
  private readonly _yolo: Yolo;
  private preprocessCanvas: HTMLCanvasElement | null = null;
  private preprocessContext: CanvasRenderingContext2D | null = null;
  private preprocessTensorData: Float32Array | null = null;
  private preprocessTensorSize = 0;
  private x0Lookup: Int32Array | null = null;
  private x1Lookup: Int32Array | null = null;
  private xWeightLookup: Float32Array | null = null;
  private y0Lookup: Int32Array | null = null;
  private y1Lookup: Int32Array | null = null;
  private yWeightLookup: Float32Array | null = null;
  private interpolationCacheKey = '';
  private topIndices: Int32Array | null = null;
  private topScores: Float32Array | null = null;
  private webGpuPipeline: any = null;
  private webGpuDevice: any = null;
  private webGpuFallbackWarned = false;

  constructor(yolo: Yolo) {
    this._yolo = yolo;
  }

  preprocessImage(img: YoloImageSource, roi: Rect | null = null): YoloPreprocessResult {
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;
    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = this._yolo.yoloOptions.imageResize ?? 'stretch';
    const { drawWidth, drawHeight, xPad, yPad, gain } =
      resizeMode === 'stretch'
        ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 }
        : this.calculateProportionalResize(sourceRect.width, sourceRect.height, modelWidth, modelHeight);
    const context = this.getPreprocessContext(modelWidth, modelHeight);
    const coversCanvas = xPad <= 0 && yPad <= 0 && drawWidth >= modelWidth && drawHeight >= modelHeight;

    if (!coversCanvas) {
      context.clearRect(0, 0, modelWidth, modelHeight);
    }
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
    const imageMean = this._yolo.yoloOptions.imageMean ?? DEFAULT_IMAGE_MEAN;
    const imageStd = this._yolo.yoloOptions.imageStd ?? DEFAULT_IMAGE_STD;

    this.writeCanvasTensor(
      imageData,
      modelWidth,
      modelHeight,
      drawWidth,
      drawHeight,
      xPad,
      yPad,
      imageMean,
      imageStd,
      tensorData,
    );

    return {
      tensorData,
      inputName: this._yolo.inputNames[0],
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

  private async preprocessImageForRun(img: YoloImageSource, roi: Rect | null = null): Promise<YoloPreprocessResult> {
    if (this._yolo.preprocessBackend !== 'webgpu') {
      return this.preprocessImage(img, roi);
    }

    try {
      return await this.preprocessImageWebGpu(img, roi);
    } catch (error) {
      if (!this.webGpuFallbackWarned) {
        console.warn('[RF-DETR] WebGPU preprocessing failed. Falling back to CPU preprocessing.', error);
        this.webGpuFallbackWarned = true;
      }

      return this.preprocessImage(img, roi);
    }
  }

  private async preprocessImageWebGpu(img: YoloImageSource, roi: Rect | null = null): Promise<YoloPreprocessResult> {
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;

    if (channels !== 3) {
      return this.preprocessImage(img, roi);
    }

    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = this._yolo.yoloOptions.imageResize ?? 'stretch';
    const { drawWidth, drawHeight, xPad, yPad, gain } =
      resizeMode === 'stretch'
        ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 }
        : this.calculateProportionalResize(sourceRect.width, sourceRect.height, modelWidth, modelHeight);
    const context = this.getPreprocessContext(modelWidth, modelHeight);
    const coversCanvas = xPad <= 0 && yPad <= 0 && drawWidth >= modelWidth && drawHeight >= modelHeight;

    if (!coversCanvas) {
      context.clearRect(0, 0, modelWidth, modelHeight);
    }

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

    const imageMean = this._yolo.yoloOptions.imageMean ?? DEFAULT_IMAGE_MEAN;
    const imageStd = this._yolo.yoloOptions.imageStd ?? DEFAULT_IMAGE_STD;
    const inputTensor = await this.createWebGpuInputTensor(context.canvas, inputShape, imageMean, imageStd);

    return {
      tensorData: new Float32Array(0),
      inputTensor,
      inputName: this._yolo.inputNames[0],
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

  private async createWebGpuInputTensor(
    canvas: HTMLCanvasElement,
    inputShape: readonly [number, number, number, number],
    imageMean: readonly [number, number, number],
    imageStd: readonly [number, number, number],
  ) {
    const [, channels, height, width] = inputShape;
    const device = await this._yolo.getWebGpuDevice();
    const usage = (globalThis as any).GPUBufferUsage;
    const textureUsage = (globalThis as any).GPUTextureUsage;
    const outputByteLength = channels * width * height * Float32Array.BYTES_PER_ELEMENT;
    const texture = device.createTexture({
      size: [width, height, 1],
      format: 'rgba8unorm',
      usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT,
    });
    const outputBuffer = device.createBuffer({
      size: outputByteLength,
      usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST,
    });
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: usage.UNIFORM | usage.COPY_DST,
    });
    const params = new Float32Array([
      width,
      height,
      0,
      0,
      1 / imageStd[0],
      1 / imageStd[1],
      1 / imageStd[2],
      0,
      -imageMean[0] / imageStd[0],
      -imageMean[1] / imageStd[1],
      -imageMean[2] / imageStd[2],
      0,
    ]);

    device.queue.copyExternalImageToTexture(
      { source: canvas },
      { texture },
      { width, height },
    );
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const pipeline = this.getWebGpuPreprocessPipeline(device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);

    texture.destroy();
    paramsBuffer.destroy();

    return this._yolo.tensorFromGpuBuffer(outputBuffer, inputShape, () => outputBuffer.destroy());
  }

  private getWebGpuPreprocessPipeline(device: any): any {
    if (this.webGpuPipeline && this.webGpuDevice === device) {
      return this.webGpuPipeline;
    }

    this.webGpuDevice = device;
    this.webGpuPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: `
struct Params {
  size: vec4<f32>,
  scale: vec4<f32>,
  bias: vec4<f32>,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(params.size.x);
  let height = u32(params.size.y);

  if (id.x >= width || id.y >= height) {
    return;
  }

  let rgba = textureLoad(inputTexture, vec2<i32>(i32(id.x), i32(id.y)), 0);
  let pixel = id.y * width + id.x;
  let planeSize = width * height;

  outputTensor[pixel] = rgba.r * params.scale.x + params.bias.x;
  outputTensor[planeSize + pixel] = rgba.g * params.scale.y + params.bias.y;
  outputTensor[planeSize * 2u + pixel] = rgba.b * params.scale.z + params.bias.z;
}
          `,
        }),
        entryPoint: 'main',
      },
    });

    return this.webGpuPipeline;
  }

  async RunObjectDetection(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<ObjectDetection[]> {
    if (this._yolo.onnxModel.modelType !== 'ObjectDetection') {
      unsupportedTask('ObjectDetection');
    }

    const input = await this.preprocessImageForRun(img, roi);
    const result = await this.runWithPreprocessedInput(input);
    const dets = result.dets?.data as Float32Array | undefined;
    const labels = result.labels?.data as Float32Array | undefined;

    if (!dets || !labels) {
      throw new Error(`Unsupported RF-DETR outputs: ${Object.keys(result).join(', ')}`);
    }

    void iou;

    return this.decodeObjectDetections(dets, labels, input, confidence);
  }

  RunObbDetection(img: YoloImageSource, confidence: number, iou: number, roi: Rect | null = null): Promise<OBBDetection[]> {
    void img;
    void confidence;
    void iou;
    void roi;
    unsupportedTask('ObbDetection');
  }

  RunSegmentation(
    img: YoloImageSource,
    confidence: number,
    pixelConfidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<Segmentation[]> {
    if (this._yolo.onnxModel.modelType !== 'Segmentation') {
      unsupportedTask('Segmentation');
    }

    void pixelConfidence;
    void iou;

    return this.runSegmentation(img, confidence, roi);
  }

  RunPoseEstimation(img: YoloImageSource, confidence: number, iou: number, roi: Rect | null = null): Promise<PoseEstimation[]> {
    void img;
    void confidence;
    void iou;
    void roi;
    unsupportedTask('PoseEstimation');
  }

  RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]> {
    void img;
    void classes;
    unsupportedTask('Classification');
  }

  private decodeObjectDetections(
    dets: Float32Array,
    logits: Float32Array,
    input: YoloPreprocessResult,
    confidence: number,
  ): ObjectDetection[] {
    const detsShape = this._yolo.onnxModel.outputShapes.dets;
    const labelsShape = this._yolo.onnxModel.outputShapes.labels;

    if (!detsShape || !labelsShape || detsShape.length !== 3 || labelsShape.length !== 3) {
      throw new Error(`Unsupported RF-DETR output shapes: ${JSON.stringify(this._yolo.onnxModel.outputShapes)}`);
    }

    const predictions = detsShape[1];
    const classCount = labelsShape[2];
    const backgroundClassIndex = this._yolo.onnxModel.labels.findIndex(label =>
      label.name.toLowerCase().startsWith(BACKGROUND_CLASS_PREFIX),
    );
    const labels = this._yolo.onnxModel.labels;
    const { topIndices, topScores, topCount } = this.getRankedCandidates(logits, predictions, classCount);
    const objects: ObjectDetection[] = [];

    for (let i = 0; i < topCount; i += 1) {
      const candidateConfidence = topScores[i];

      if (candidateConfidence < confidence) {
        continue;
      }

      const flatIndex = topIndices[i];
      const prediction = Math.trunc(flatIndex / classCount);
      const labelIndex = flatIndex - prediction * classCount;

      if (labelIndex === backgroundClassIndex) {
        continue;
      }

      const label = labels[labelIndex];

      if (!label) {
        continue;
      }

      const boxOffset = prediction * 4;
      const cx = dets[boxOffset];
      const cy = dets[boxOffset + 1];
      const w = dets[boxOffset + 2];
      const h = dets[boxOffset + 3];
      const x1 = cx - w / 2;
      const y1 = cy - h / 2;
      const x2 = cx + w / 2;
      const y2 = cy + h / 2;

      objects.push(new ObjectDetection({
        label,
        confidence: candidateConfidence,
        boundingBox: this.scaleNormalizedBoundingBox(x1, y1, x2, y2, input),
      }));
    }

    return objects;
  }

  private async runSegmentation(img: YoloImageSource, confidence: number, roi: Rect | null): Promise<Segmentation[]> {
    const input = await this.preprocessImageForRun(img, roi);
 
    const result = await this.runWithPreprocessedInput(input);
 
    
    const dets = result.dets?.data as Float32Array | undefined;
    const logits = result.labels?.data as Float32Array | undefined;
    const masks = result.masks?.data as Float32Array | undefined;

    if (!dets || !logits || !masks) {
      throw new Error(`Unsupported RF-DETR segmentation outputs: ${Object.keys(result).join(', ')}`);
    }

     return this.decodeSegmentations(dets, logits, masks, input, confidence);
  }

  private async runWithPreprocessedInput(input: YoloPreprocessResult) {
    const inputTensor = input.inputTensor ?? this._yolo.tensor('float32', input.tensorData, input.inputShape);

    try {
      return await this._yolo.run({
        [input.inputName]: inputTensor,
      });
    } finally {
      input.inputTensor?.dispose();
    }
  }

  private decodeSegmentations(
    dets: Float32Array,
    logits: Float32Array,
    masks: Float32Array,
    input: YoloPreprocessResult,
    confidence: number,
  ): Segmentation[] {
    const detsShape = this._yolo.onnxModel.outputShapes.dets;
    const labelsShape = this._yolo.onnxModel.outputShapes.labels;
    const masksShape = this._yolo.onnxModel.outputShapes.masks;

    if (
      !detsShape ||
      !labelsShape ||
      !masksShape ||
      detsShape.length !== 3 ||
      labelsShape.length !== 3 ||
      masksShape.length !== 4
    ) {
      throw new Error(`Unsupported RF-DETR segmentation output shapes: ${JSON.stringify(this._yolo.onnxModel.outputShapes)}`);
    }

    const predictions = detsShape[1];
    const classCount = labelsShape[2];
    const maskHeight = masksShape[2];
    const maskWidth = masksShape[3];
    const maskPlaneSize = maskWidth * maskHeight;
    const backgroundClassIndex = this._yolo.onnxModel.labels.findIndex(label =>
      label.name.toLowerCase().startsWith(BACKGROUND_CLASS_PREFIX),
    );
    const labels = this._yolo.onnxModel.labels;
 

    const { topIndices, topScores, topCount } = this.getRankedCandidates(logits, predictions, classCount);
    const segmentations: Segmentation[] = [];
 
    for (let i = 0; i < topCount; i += 1) {
      const candidateConfidence = topScores[i];

      if (candidateConfidence < confidence) {
        continue;
      }

      const flatIndex = topIndices[i];
      const prediction = Math.trunc(flatIndex / classCount);
      const labelIndex = flatIndex - prediction * classCount;

      if (labelIndex === backgroundClassIndex) {
        continue;
      }

      const label = labels[labelIndex];

      if (!label) {
        continue;
      }

      const boxOffset = prediction * 4;
      const cx = dets[boxOffset];
      const cy = dets[boxOffset + 1];
      const w = dets[boxOffset + 2];
      const h = dets[boxOffset + 3];
      const x1 = cx - w / 2;
      const y1 = cy - h / 2;
      const x2 = cx + w / 2;
      const y2 = cy + h / 2;
      const boundingBox = this.scaleNormalizedBoundingBox(x1, y1, x2, y2, input);
      const bitPackedPixelMask = this.packRfdetrMask(
        masks,
        prediction * maskPlaneSize,
        maskWidth,
        maskHeight,
        boundingBox,
        input,
      );

      segmentations.push(new Segmentation({
        label,
        confidence: candidateConfidence,
        boundingBox,
        bitPackedPixelMask,
      }));
    }

    return segmentations;
  }

  private getRankedCandidates(
    logits: Float32Array,
    predictions: number,
    classCount: number,
  ): { topIndices: Int32Array; topScores: Float32Array; topCount: number } {
    const maxDetections = predictions;
    const { topIndices, topScores } = this.getTopKBuffers(maxDetections);
    let topCount = 0;
    let minScore = Number.POSITIVE_INFINITY;
    let minPosition = -1;

    for (let prediction = 0; prediction < predictions; prediction += 1) {
      const labelOffset = prediction * classCount;

      for (let labelIndex = 0; labelIndex < classCount; labelIndex += 1) {
        const score = this.sigmoid(logits[labelOffset + labelIndex]);
        const flatIndex = labelOffset + labelIndex;

        if (topCount < maxDetections) {
          topIndices[topCount] = flatIndex;
          topScores[topCount] = score;

          if (score < minScore) {
            minScore = score;
            minPosition = topCount;
          }

          topCount += 1;
          continue;
        }

        if (score <= minScore) {
          continue;
        }

        topIndices[minPosition] = flatIndex;
        topScores[minPosition] = score;
        minScore = topScores[0];
        minPosition = 0;

        for (let i = 1; i < topCount; i += 1) {
          const candidateScore = topScores[i];

          if (candidateScore < minScore) {
            minScore = candidateScore;
            minPosition = i;
          }
        }
      }
    }

    this.sortTopKDescending(topIndices, topScores, topCount);

    return { topIndices, topScores, topCount };
  }

  private packRfdetrMask(
    masks: Float32Array,
    maskOffset: number,
    maskWidth: number,
    maskHeight: number,
    box: Rect,
    input: YoloPreprocessResult,
  ): Uint8Array {
    const targetWidth = box.right - box.left;
    const targetHeight = box.bottom - box.top;

    if (targetWidth <= 0 || targetHeight <= 0 || maskWidth <= 0 || maskHeight <= 0) {
      return new Uint8Array();
    }

    const totalPixels = targetWidth * targetHeight;
    const packed = new Uint8Array(Math.ceil(totalPixels / 8));
    let cropLeft = 0;
    let cropTop = 0;
    let cropRight = maskWidth;
    let cropBottom = maskHeight;

    if (input.resizeMode !== 'stretch') {
      const inputWidth = input.inputShape[3];
      const inputHeight = input.inputShape[2];
      const scale = Math.min(inputWidth / input.sourceWidth, inputHeight / input.sourceHeight);
      const scaledWidth = Math.trunc(input.sourceWidth * scale);
      const scaledHeight = Math.trunc(input.sourceHeight * scale);
      const padX = (inputWidth - scaledWidth) / 2;
      const padY = (inputHeight - scaledHeight) / 2;

      cropLeft = clamp(Math.round(padX * maskWidth / inputWidth), 0, maskWidth - 1);
      cropTop = clamp(Math.round(padY * maskHeight / inputHeight), 0, maskHeight - 1);
      cropRight = clamp(Math.round((padX + scaledWidth) * maskWidth / inputWidth), cropLeft + 1, maskWidth);
      cropBottom = clamp(Math.round((padY + scaledHeight) * maskHeight / inputHeight), cropTop + 1, maskHeight);
    }

    const cropWidth = cropRight - cropLeft;
    const cropHeight = cropBottom - cropTop;
    const xScale = cropWidth / input.sourceWidth;
    const yScale = cropHeight / input.sourceHeight;

    for (let y = 0; y < targetHeight; y += 1) {
      const absoluteY = box.top + y;
      const sourceY = (absoluteY + 0.5) * yScale - 0.5;
      const y0Local = clamp(Math.floor(sourceY), 0, cropHeight - 1);
      const y1Local = y0Local < cropHeight - 1 ? y0Local + 1 : y0Local;
      const yWeight = sourceY - y0Local;
      const row0 = maskOffset + (cropTop + y0Local) * maskWidth;
      const row1 = maskOffset + (cropTop + y1Local) * maskWidth;
      const targetRow = y * targetWidth;

      for (let x = 0; x < targetWidth; x += 1) {
        const absoluteX = box.left + x;
        const sourceX = (absoluteX + 0.5) * xScale - 0.5;
        const x0Local = clamp(Math.floor(sourceX), 0, cropWidth - 1);
        const x1Local = x0Local < cropWidth - 1 ? x0Local + 1 : x0Local;
        const xWeight = sourceX - x0Local;
        const x0 = cropLeft + x0Local;
        const x1 = cropLeft + x1Local;
        const topLeft = masks[row0 + x0];
        const topRight = masks[row0 + x1];
        const bottomLeft = masks[row1 + x0];
        const bottomRight = masks[row1 + x1];
        const top = topLeft + (topRight - topLeft) * xWeight;
        const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;

        // Official RF-DETR inference thresholds masks at > 0 after optional resize/crop.
        if (top + (bottom - top) * yWeight > 0) {
          const pixelIndex = targetRow + x;
          packed[pixelIndex >> 3] |= 1 << (pixelIndex & 0b0111);
        }
      }
    }

    return packed;
  }

  private getTopKBuffers(size: number): { topIndices: Int32Array; topScores: Float32Array } {
    if (!this.topIndices || this.topIndices.length < size) {
      this.topIndices = new Int32Array(size);
      this.topScores = new Float32Array(size);
    }

    return {
      topIndices: this.topIndices,
      topScores: this.topScores!,
    };
  }

  private sortTopKDescending(indices: Int32Array, scores: Float32Array, length: number): void {
    for (let i = 1; i < length; i += 1) {
      const score = scores[i];
      const index = indices[i];
      let j = i - 1;

      while (j >= 0 && scores[j] < score) {
        scores[j + 1] = scores[j];
        indices[j + 1] = indices[j];
        j -= 1;
      }

      scores[j + 1] = score;
      indices[j + 1] = index;
    }
  }

  private sigmoid(value: number): number {
    return 1 / (1 + Math.exp(-value));
  }

  private scaleNormalizedBoundingBox(x1: number, y1: number, x2: number, y2: number, input: YoloPreprocessResult): Rect {
    const roiLeft = input.roi?.left ?? 0;
    const roiTop = input.roi?.top ?? 0;

    if (input.resizeMode === 'stretch') {
      return {
        left: roiLeft + clamp(Math.trunc(x1 * input.sourceWidth), 0, input.sourceWidth - 1),
        top: roiTop + clamp(Math.trunc(y1 * input.sourceHeight), 0, input.sourceHeight - 1),
        right: roiLeft + clamp(Math.trunc(x2 * input.sourceWidth), 0, input.sourceWidth),
        bottom: roiTop + clamp(Math.trunc(y2 * input.sourceHeight), 0, input.sourceHeight),
      };
    }

    return this._yolo.scaleBoundingBox(
      x1 * input.inputShape[3],
      y1 * input.inputShape[2],
      x2 * input.inputShape[3],
      y2 * input.inputShape[2],
      input,
    );
  }

  private getInputShape(): readonly [number, number, number, number] {
    const shape = Object.values(this._yolo.onnxModel.inputShapes)[0];

    if (!shape || shape.length !== 4) {
      throw new Error(`Unsupported RF-DETR input shape: ${JSON.stringify(shape)}`);
    }

    return [shape[0], shape[1], shape[2], shape[3]] as const;
  }

  private writeCanvasTensor(
    imageData: Uint8ClampedArray,
    modelWidth: number,
    modelHeight: number,
    drawWidth: number,
    drawHeight: number,
    xPad: number,
    yPad: number,
    imageMean: readonly [number, number, number],
    imageStd: readonly [number, number, number],
    tensorData: Float32Array,
  ): void {
    const pixelCount = modelWidth * modelHeight;
    const redScale = 1 / (255 * imageStd[0]);
    const greenScale = 1 / (255 * imageStd[1]);
    const blueScale = 1 / (255 * imageStd[2]);
    const redBias = -imageMean[0] / imageStd[0];
    const greenBias = -imageMean[1] / imageStd[1];
    const blueBias = -imageMean[2] / imageStd[2];
    const left = Math.max(0, Math.trunc(xPad));
    const top = Math.max(0, Math.trunc(yPad));
    const right = Math.min(modelWidth, Math.ceil(xPad + drawWidth));
    const bottom = Math.min(modelHeight, Math.ceil(yPad + drawHeight));
    const canUseUint32 = (imageData.byteOffset & 3) === 0 && (imageData.byteLength & 3) === 0;
    const uint32Pixels = canUseUint32
      ? new Uint32Array(imageData.buffer, imageData.byteOffset, pixelCount)
      : null;

    if (left === 0 && top === 0 && right === modelWidth && bottom === modelHeight) {
      if (uint32Pixels) {
        for (let pixel = 0; pixel < pixelCount; pixel += 1) {
          const rgba = uint32Pixels[pixel];
          tensorData[pixel] = (rgba & 0xff) * redScale + redBias;
          tensorData[pixelCount + pixel] = ((rgba >>> 8) & 0xff) * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = ((rgba >>> 16) & 0xff) * blueScale + blueBias;
        }

        return;
      }

      for (let imageOffset = 0, pixel = 0; pixel < pixelCount; imageOffset += 4, pixel += 1) {
        tensorData[pixel] = imageData[imageOffset] * redScale + redBias;
        tensorData[pixelCount + pixel] = imageData[imageOffset + 1] * greenScale + greenBias;
        tensorData[pixelCount * 2 + pixel] = imageData[imageOffset + 2] * blueScale + blueBias;
      }

      return;
    }

    tensorData.fill(0);

    for (let y = top; y < bottom; y += 1) {
      let imageOffset = y * modelWidth + left;
      let pixel = y * modelWidth + left;

      if (uint32Pixels) {
        for (let x = left; x < right; x += 1, imageOffset += 1, pixel += 1) {
          const rgba = uint32Pixels[imageOffset];
          tensorData[pixel] = (rgba & 0xff) * redScale + redBias;
          tensorData[pixelCount + pixel] = ((rgba >>> 8) & 0xff) * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = ((rgba >>> 16) & 0xff) * blueScale + blueBias;
        }
      } else {
        let byteOffset = imageOffset * 4;

        for (let x = left; x < right; x += 1, byteOffset += 4, pixel += 1) {
          tensorData[pixel] = imageData[byteOffset] * redScale + redBias;
          tensorData[pixelCount + pixel] = imageData[byteOffset + 1] * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = imageData[byteOffset + 2] * blueScale + blueBias;
        }
      }
    }
  }

  private writePreprocessedTensor(
    imageData: Uint8ClampedArray,
    sourceWidth: number,
    sourceHeight: number,
    modelWidth: number,
    modelHeight: number,
    drawWidth: number,
    drawHeight: number,
    xPad: number,
    yPad: number,
    imageMean: readonly [number, number, number],
    imageStd: readonly [number, number, number],
    tensorData: Float32Array,
  ): void {
    const pixelCount = modelWidth * modelHeight;
    tensorData.fill(0);

    const outputLeft = Math.trunc(xPad);
    const outputTop = Math.trunc(yPad);
    const outputWidth = Math.max(1, Math.trunc(drawWidth));
    const outputHeight = Math.max(1, Math.trunc(drawHeight));
    this.ensureInterpolationCache(sourceWidth, sourceHeight, outputWidth, outputHeight);
    const x0Lookup = this.x0Lookup!;
    const x1Lookup = this.x1Lookup!;
    const xWeightLookup = this.xWeightLookup!;
    const y0Lookup = this.y0Lookup!;
    const y1Lookup = this.y1Lookup!;
    const yWeightLookup = this.yWeightLookup!;

    for (let y = 0; y < outputHeight; y += 1) {
      const targetY = outputTop + y;

      if (targetY < 0 || targetY >= modelHeight) {
        continue;
      }

      const y0 = y0Lookup[y];
      const y1 = y1Lookup[y];
      const yWeight = yWeightLookup[y];
      const topRow = y0 * sourceWidth * 4;
      const bottomRow = y1 * sourceWidth * 4;

      for (let x = 0; x < outputWidth; x += 1) {
        const targetX = outputLeft + x;

        if (targetX < 0 || targetX >= modelWidth) {
          continue;
        }

        const x0 = x0Lookup[x];
        const x1 = x1Lookup[x];
        const xWeight = xWeightLookup[x];
        const targetPixel = targetY * modelWidth + targetX;
        const topLeft = topRow + x0 * 4;
        const topRight = topRow + x1 * 4;
        const bottomLeft = bottomRow + x0 * 4;
        const bottomRight = bottomRow + x1 * 4;
        const r = this.interpolate(imageData[topLeft], imageData[topRight], imageData[bottomLeft], imageData[bottomRight], xWeight, yWeight);
        const g = this.interpolate(imageData[topLeft + 1], imageData[topRight + 1], imageData[bottomLeft + 1], imageData[bottomRight + 1], xWeight, yWeight);
        const b = this.interpolate(imageData[topLeft + 2], imageData[topRight + 2], imageData[bottomLeft + 2], imageData[bottomRight + 2], xWeight, yWeight);

        tensorData[targetPixel] = (r / 255 - imageMean[0]) / imageStd[0];
        tensorData[pixelCount + targetPixel] = (g / 255 - imageMean[1]) / imageStd[1];
        tensorData[pixelCount * 2 + targetPixel] = (b / 255 - imageMean[2]) / imageStd[2];
      }
    }
  }

  private ensureInterpolationCache(sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number): void {
    const cacheKey = `${sourceWidth}:${sourceHeight}:${outputWidth}:${outputHeight}`;

    if (this.interpolationCacheKey === cacheKey) {
      return;
    }

    this.interpolationCacheKey = cacheKey;
    this.x0Lookup = new Int32Array(outputWidth);
    this.x1Lookup = new Int32Array(outputWidth);
    this.xWeightLookup = new Float32Array(outputWidth);
    this.y0Lookup = new Int32Array(outputHeight);
    this.y1Lookup = new Int32Array(outputHeight);
    this.yWeightLookup = new Float32Array(outputHeight);

    this.fillInterpolationAxis(this.x0Lookup, this.x1Lookup, this.xWeightLookup, outputWidth, sourceWidth);
    this.fillInterpolationAxis(this.y0Lookup, this.y1Lookup, this.yWeightLookup, outputHeight, sourceHeight);
  }

  private fillInterpolationAxis(
    lowerLookup: Int32Array,
    upperLookup: Int32Array,
    weightLookup: Float32Array,
    targetSize: number,
    sourceSize: number,
  ): void {
    for (let target = 0; target < targetSize; target += 1) {
      const source = (target + 0.5) * (sourceSize / targetSize) - 0.5;
      const index = Math.floor(source);

      if (index < 0) {
        lowerLookup[target] = 0;
        upperLookup[target] = 0;
        weightLookup[target] = 0;
      } else if (index >= sourceSize - 1) {
        lowerLookup[target] = sourceSize - 1;
        upperLookup[target] = sourceSize - 1;
        weightLookup[target] = 0;
      } else {
        lowerLookup[target] = index;
        upperLookup[target] = index + 1;
        weightLookup[target] = source - index;
      }
    }
  }

  private interpolate(topLeft: number, topRight: number, bottomLeft: number, bottomRight: number, xWeight: number, yWeight: number): number {
    const top = topLeft + (topRight - topLeft) * xWeight;
    const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
    return top + (bottom - top) * yWeight;
  }

  private getSourceRect(img: YoloImageSource, roi: Rect | null): { x: number; y: number; width: number; height: number } {
    const { width, height } = this.getImageSourceSize(img);

    if (!roi) {
      return { x: 0, y: 0, width, height };
    }

    const left = clamp(Math.trunc(roi.left), 0, width - 1);
    const top = clamp(Math.trunc(roi.top), 0, height - 1);
    const right = clamp(Math.trunc(roi.right), left + 1, width);
    const bottom = clamp(Math.trunc(roi.bottom), top + 1, height);

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
}
