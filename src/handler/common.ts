import type { Detection, KeyPoint, LabelModel, Rect, YoloPreprocessResult } from '../types';

export interface DecodedObject {
  label: LabelModel;
  confidence: number;
  boundingBox: Rect;
  boundingBoxUnscaled: Rect;
  boundingBoxIndex: number;
  orientationAngle?: number;
  keyPoints?: KeyPoint[];
  bitPackedPixelMask?: Uint8Array;
}

export function unsupportedTask(task: string): never {
  throw new Error(`${task} is not supported by this YOLO model.`);
}

export function toDetection(object: DecodedObject): Detection {
  return {
    label: object.label,
    confidence: object.confidence,
    boundingBox: object.boundingBox,
  };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

export function scalePoint(x: number, y: number, input: YoloPreprocessResult): { x: number; y: number } {
  const roiLeft = input.roi?.left ?? 0;
  const roiTop = input.roi?.top ?? 0;

  return {
    x: roiLeft + clamp(Math.trunc((x - input.xPad) * input.gain), 0, input.sourceWidth - 1),
    y: roiTop + clamp(Math.trunc((y - input.yPad) * input.gain), 0, input.sourceHeight - 1),
  };
}

export function removeOverlappingBoxes<T extends { confidence: number; boundingBox: Rect }>(
  detections: T[],
  iouThreshold: number,
): T[] {
  if (detections.length === 0 || iouThreshold <= 0) {
    return detections;
  }

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const results: T[] = [];

  for (const detection of sorted) {
    const overlaps = results.some(item => calculateIoU(detection.boundingBox, item.boundingBox) > iouThreshold);

    if (!overlaps) {
      results.push(detection);
    }
  }

  return results;
}

export function calculateIoU(a: Rect, b: Rect): number {
  const left = Math.max(a.left, b.left);
  const top = Math.max(a.top, b.top);
  const right = Math.min(a.right, b.right);
  const bottom = Math.min(a.bottom, b.bottom);
  const width = right - left;
  const height = bottom - top;

  if (width <= 0 || height <= 0) {
    return 0;
  }

  const intersection = width * height;
  const areaA = (a.right - a.left) * (a.bottom - a.top);
  const areaB = (b.right - b.left) * (b.bottom - b.top);

  return intersection / (areaA + areaB - intersection);
}

export function packSegmentationMask(
  proto: Float32Array,
  maskWeights: ArrayLike<number>,
  maskWidth: number,
  maskHeight: number,
  crop: Rect,
  targetWidth: number,
  targetHeight: number,
  pixelConfidence: number,
): Uint8Array {
  if (targetWidth <= 0 || targetHeight <= 0 || maskWidth <= 0 || maskHeight <= 0) {
    return new Uint8Array();
  }

  const totalPixels = targetWidth * targetHeight;
  const packed = new Uint8Array(Math.ceil(totalPixels / 8));
  const planeSize = maskWidth * maskHeight;
  const cropLeft = clamp(Math.trunc(crop.left), 0, maskWidth - 1);
  const cropTop = clamp(Math.trunc(crop.top), 0, maskHeight - 1);
  const cropRight = clamp(Math.trunc(crop.right), cropLeft + 1, maskWidth);
  const cropBottom = clamp(Math.trunc(crop.bottom), cropTop + 1, maskHeight);
  const cropWidth = cropRight - cropLeft;
  const cropHeight = cropBottom - cropTop;
  const activeWeights: number[] = [];
  const activeOffsets: number[] = [];

  for (let channel = 0; channel < maskWeights.length; channel += 1) {
    const weight = maskWeights[channel] ?? 0;

    if (weight === 0) {
      continue;
    }

    activeWeights.push(weight);
    activeOffsets.push(channel * planeSize);
  }

  if (activeWeights.length === 0) {
    return packed;
  }

  const lowResMask = new Float32Array(cropWidth * cropHeight);

  for (let y = 0; y < cropHeight; y += 1) {
    const protoRowOffset = (cropTop + y) * maskWidth + cropLeft;
    const maskRowOffset = y * cropWidth;

    for (let x = 0; x < cropWidth; x += 1) {
      const protoOffset = protoRowOffset + x;
      let pixelWeight = 0;

      for (let channel = 0; channel < activeWeights.length; channel += 1) {
        pixelWeight += (proto[activeOffsets[channel] + protoOffset] ?? 0) * activeWeights[channel];
      }

      lowResMask[maskRowOffset + x] = sigmoid(pixelWeight);
    }
  }

  if (targetWidth === cropWidth && targetHeight === cropHeight) {
    for (let pixelIndex = 0; pixelIndex < totalPixels; pixelIndex += 1) {
      if (lowResMask[pixelIndex] > pixelConfidence) {
        packed[pixelIndex >> 3] |= 1 << (pixelIndex & 0b0111);
      }
    }

    return packed;
  }

  const x0Lookup = new Int32Array(targetWidth);
  const x1Lookup = new Int32Array(targetWidth);
  const xWeightLookup = new Float32Array(targetWidth);
  const xScale = cropWidth / targetWidth;
  const yScale = cropHeight / targetHeight;

  for (let targetX = 0; targetX < targetWidth; targetX += 1) {
    const sourceX = (targetX + 0.5) * xScale - 0.5;
    const x0 = clamp(Math.floor(sourceX), 0, cropWidth - 1);
    const x1 = x0 < cropWidth - 1 ? x0 + 1 : x0;

    x0Lookup[targetX] = x0;
    x1Lookup[targetX] = x1;
    xWeightLookup[targetX] = sourceX - x0;
  }

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = (targetY + 0.5) * yScale - 0.5;
    const y0 = clamp(Math.floor(sourceY), 0, cropHeight - 1);
    const y1 = y0 < cropHeight - 1 ? y0 + 1 : y0;
    const yWeight = sourceY - y0;
    const row0 = y0 * cropWidth;
    const row1 = y1 * cropWidth;
    const targetRow = targetY * targetWidth;

    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const x0 = x0Lookup[targetX];
      const x1 = x1Lookup[targetX];
      const xWeight = xWeightLookup[targetX];
      const topLeft = lowResMask[row0 + x0];
      const topRight = lowResMask[row0 + x1];
      const bottomLeft = lowResMask[row1 + x0];
      const bottomRight = lowResMask[row1 + x1];
      const top = topLeft + (topRight - topLeft) * xWeight;
      const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;

      if (top + (bottom - top) * yWeight > pixelConfidence) {
        const pixelIndex = targetRow + targetX;
        packed[pixelIndex >> 3] |= 1 << (pixelIndex & 0b0111);
      }
    }
  }

  return packed;
}

export function downscaleBoxToMask(box: Rect, maskWidth: number, maskHeight: number, inputWidth: number, inputHeight: number): Rect {
  const scalingFactorW = maskWidth / inputWidth;
  const scalingFactorH = maskHeight / inputHeight;

  return {
    left: clamp(Math.floor(box.left * scalingFactorW), 0, maskWidth - 1),
    top: clamp(Math.floor(box.top * scalingFactorH), 0, maskHeight - 1),
    right: clamp(Math.ceil(box.right * scalingFactorW), 0, maskWidth - 1),
    bottom: clamp(Math.ceil(box.bottom * scalingFactorH), 0, maskHeight - 1),
  };
}
