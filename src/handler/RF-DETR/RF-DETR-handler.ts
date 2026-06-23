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
    const context = this.getPreprocessContext(sourceRect.width, sourceRect.height);

    context.clearRect(0, 0, sourceRect.width, sourceRect.height);
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      0,
      0,
      sourceRect.width,
      sourceRect.height,
    );

    const imageData = context.getImageData(0, 0, sourceRect.width, sourceRect.height).data;
    const pixelCount = modelWidth * modelHeight;
    const tensorData = this.getPreprocessTensorData(channels * pixelCount);
    const imageMean = this._yolo.yoloOptions.imageMean ?? DEFAULT_IMAGE_MEAN;
    const imageStd = this._yolo.yoloOptions.imageStd ?? DEFAULT_IMAGE_STD;

    this.writePreprocessedTensor(
      imageData,
      sourceRect.width,
      sourceRect.height,
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

  async RunObjectDetection(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<ObjectDetection[]> {
    if (this._yolo.onnxModel.modelType !== 'ObjectDetection') {
      unsupportedTask('ObjectDetection');
    }

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
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
    void img;
    void confidence;
    void pixelConfidence;
    void iou;
    void roi;
    unsupportedTask('Segmentation');
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
    const labels = backgroundClassIndex >= 0
      ? this._yolo.onnxModel.labels
          .filter(label => label.index !== backgroundClassIndex)
          .map((label, index) => ({ ...label, index }))
      : this._yolo.onnxModel.labels;
    const maxDetections = predictions;
    const { topIndices, topScores } = this.getTopKBuffers(maxDetections);
    const objects: ObjectDetection[] = [];
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

      const mappedLabelIndex = backgroundClassIndex >= 0 && labelIndex > backgroundClassIndex
        ? labelIndex - 1
        : labelIndex;
      const label = labels[mappedLabelIndex];

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
