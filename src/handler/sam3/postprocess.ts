import { Segmentation } from '../../types';
import type { Rect } from '../../types';
import { DrawTool } from '../../draw-tool';
import { clamp, cxcywhToRect, sigmoid } from './preprocess';
import { SAM3_MASK_SIZE } from './preprocess';

export function packBinaryMask(mask: Uint8Array | Float32Array, width: number, height: number, threshold = 0.5): Uint8Array {
  const total = width * height;
  const packed = new Uint8Array(Math.ceil(total / 8));

  for (let index = 0; index < total; index += 1) {
    if ((mask[index] ?? 0) > threshold) {
      packed[index >> 3] |= 1 << (index & 0b0111);
    }
  }

  return packed;
}

export function bilinearResize(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const target = new Float32Array(targetWidth * targetHeight);
  const xScale = sourceWidth / targetWidth;
  const yScale = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = (y + 0.5) * yScale - 0.5;
    const y0 = clamp(Math.floor(sourceY), 0, sourceHeight - 1);
    const y1 = y0 < sourceHeight - 1 ? y0 + 1 : y0;
    const wy = sourceY - y0;

    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = (x + 0.5) * xScale - 0.5;
      const x0 = clamp(Math.floor(sourceX), 0, sourceWidth - 1);
      const x1 = x0 < sourceWidth - 1 ? x0 + 1 : x0;
      const wx = sourceX - x0;
      const v00 = source[y0 * sourceWidth + x0] ?? 0;
      const v01 = source[y0 * sourceWidth + x1] ?? 0;
      const v10 = source[y1 * sourceWidth + x0] ?? 0;
      const v11 = source[y1 * sourceWidth + x1] ?? 0;
      target[y * targetWidth + x] =
        v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy;
    }
  }

  return target;
}

export function maskBounds(mask: Float32Array | Uint8Array, width: number, height: number, threshold = 0.5): Rect {
  let left = width;
  let top = height;
  let right = 0;
  let bottom = 0;

  for (let y = 0; y < height; y += 1) {
    const row = y * width;

    for (let x = 0; x < width; x += 1) {
      if ((mask[row + x] ?? 0) > threshold) {
        if (x < left) left = x;
        if (y < top) top = y;
        if (x + 1 > right) right = x + 1;
        if (y + 1 > bottom) bottom = y + 1;
      }
    }
  }

  if (right <= left || bottom <= top) {
    return { left: 0, top: 0, right: width, bottom: height };
  }

  return { left, top, right, bottom };
}

export function packCroppedBinaryMask(
  mask: Float32Array | Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  crop: Rect,
  threshold = 0.5,
): Uint8Array {
  const width = crop.right - crop.left;
  const height = crop.bottom - crop.top;
  const packed = new Uint8Array(Math.ceil(Math.max(0, width * height) / 8));

  for (let y = 0; y < height; y += 1) {
    const sourceY = crop.top + y;

    if (sourceY < 0 || sourceY >= sourceHeight) {
      continue;
    }

    const sourceRow = sourceY * sourceWidth;

    for (let x = 0; x < width; x += 1) {
      const sourceX = crop.left + x;

      if (sourceX < 0 || sourceX >= sourceWidth) {
        continue;
      }

      if ((mask[sourceRow + sourceX] ?? 0) > threshold) {
        const pixelIndex = y * width + x;
        packed[pixelIndex >> 3] |= 1 << (pixelIndex & 0b0111);
      }
    }
  }

  return packed;
}

function integerRect(rect: Rect, sourceWidth: number, sourceHeight: number): Rect {
  const left = clamp(Math.floor(rect.left), 0, sourceWidth);
  const top = clamp(Math.floor(rect.top), 0, sourceHeight);
  const right = clamp(Math.ceil(rect.right), left, sourceWidth);
  const bottom = clamp(Math.ceil(rect.bottom), top, sourceHeight);

  return {
    left,
    top,
    right: right > left ? right : Math.min(sourceWidth, left + 1),
    bottom: bottom > top ? bottom : Math.min(sourceHeight, top + 1),
  };
}

function toSegmentation(
  probabilities: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  label: string,
  confidence: number,
  pixelThreshold: number,
  boundingBox?: Rect,
): Segmentation {
  const box = integerRect(boundingBox ?? maskBounds(probabilities, sourceWidth, sourceHeight, pixelThreshold), sourceWidth, sourceHeight);
  const packed = packCroppedBinaryMask(probabilities, sourceWidth, sourceHeight, box, pixelThreshold);
  const segmentation = new Segmentation({
    label: { index: 0, name: label },
    confidence,
    boundingBox: box,
    bitPackedPixelMask: packed,
  });
  segmentation.segmentationEdgePoints = DrawTool.extractSegmentationEdgePoints(segmentation);
  return segmentation;
}

export function logitsToSegmentation(
  logits: Float32Array,
  maskWidth: number,
  maskHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  label: string,
  confidence: number,
  pixelThreshold = 0.5,
  applySigmoid = true,
  boundingBox?: Rect,
): Segmentation {
  const resized = bilinearResize(logits, maskWidth, maskHeight, sourceWidth, sourceHeight);
  const probabilities = applySigmoid
    ? Float32Array.from(resized, value => sigmoid(value))
    : resized;

  return toSegmentation(
    probabilities,
    sourceWidth,
    sourceHeight,
    label,
    confidence,
    pixelThreshold,
    boundingBox,
  );
}

export function decodePcsOutputs(
  predMasks: Float32Array,
  predBoxes: Float32Array,
  predLogits: Float32Array,
  presenceLogits: Float32Array,
  maskShape: readonly number[],
  boxShape: readonly number[],
  sourceWidth: number,
  sourceHeight: number,
  text: string,
  threshold: number,
  pixelThreshold: number,
): Segmentation[] {
  const queryCount = predLogits.length;
  const maskHeight = maskShape[maskShape.length - 2] ?? SAM3_MASK_SIZE;
  const maskWidth = maskShape[maskShape.length - 1] ?? SAM3_MASK_SIZE;
  const boxStride = boxShape[boxShape.length - 1] ?? 4;
  const presence = sigmoid(presenceLogits[0] ?? 0);
  const results: Segmentation[] = [];

  for (let query = 0; query < queryCount; query += 1) {
    const score = sigmoid(predLogits[query] ?? 0) * presence;

    if (score <= threshold) {
      continue;
    }

    const maskOffset = query * maskHeight * maskWidth;
    const mask = predMasks.subarray(maskOffset, maskOffset + maskHeight * maskWidth);
    const boxOffset = query * boxStride;
    const boundingBox =
      boxStride >= 4
        ? cxcywhToRect(
            predBoxes[boxOffset] ?? 0,
            predBoxes[boxOffset + 1] ?? 0,
            predBoxes[boxOffset + 2] ?? 0,
            predBoxes[boxOffset + 3] ?? 0,
            sourceWidth,
            sourceHeight,
          )
        : undefined;
    results.push(
      logitsToSegmentation(
        mask,
        maskWidth,
        maskHeight,
        sourceWidth,
        sourceHeight,
        text || 'object',
        score,
        pixelThreshold,
        true,
        boundingBox,
      ),
    );
  }

  return results;
}

export function copyLowResMask(data: Float32Array, index: number, width: number, height: number): Float32Array {
  const size = width * height;
  return data.slice(index * size, (index + 1) * size);
}
