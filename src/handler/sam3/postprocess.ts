import { Segmentation } from '../../types';
import type { Point, Rect } from '../../types';
import { DrawTool } from '../../draw-tool';
import { clamp, cxcywhToRect, sigmoid } from './preprocess';
import { SAM3_MASK_SIZE } from './preprocess';

/** Cap per-instance mask raster so 4K/8K photos don't bilinear-upsample 288 → full image. */
export const SAM3_MAX_OUTPUT_MASK_SIDE = 576;

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

export function bilinearResizeRegion(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  srcLeft: number,
  srcTop: number,
  srcRight: number,
  srcBottom: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const target = new Float32Array(targetWidth * targetHeight);
  const regionWidth = Math.max(srcRight - srcLeft, 1e-6);
  const regionHeight = Math.max(srcBottom - srcTop, 1e-6);

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = srcTop + ((y + 0.5) / targetHeight) * regionHeight - 0.5;
    const y0 = clamp(Math.floor(sourceY), 0, sourceHeight - 1);
    const y1 = y0 < sourceHeight - 1 ? y0 + 1 : y0;
    const wy = sourceY - y0;

    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = srcLeft + ((x + 0.5) / targetWidth) * regionWidth - 0.5;
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

export function bilinearResize(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  return bilinearResizeRegion(source, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight, targetWidth, targetHeight);
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

function logitThreshold(pixelThreshold: number, applySigmoid: boolean): number {
  if (!applySigmoid) {
    return pixelThreshold;
  }

  const probability = clamp(pixelThreshold, 1e-6, 1 - 1e-6);
  return Math.log(probability / (1 - probability));
}

function sourceRectToMaskRect(
  box: Rect,
  sourceWidth: number,
  sourceHeight: number,
  maskWidth: number,
  maskHeight: number,
): Rect {
  return {
    left: (box.left / sourceWidth) * maskWidth,
    top: (box.top / sourceHeight) * maskHeight,
    right: (box.right / sourceWidth) * maskWidth,
    bottom: (box.bottom / sourceHeight) * maskHeight,
  };
}

function maskRectToSourceRect(
  box: Rect,
  maskWidth: number,
  maskHeight: number,
  sourceWidth: number,
  sourceHeight: number,
): Rect {
  return {
    left: (box.left / maskWidth) * sourceWidth,
    top: (box.top / maskHeight) * sourceHeight,
    right: (box.right / maskWidth) * sourceWidth,
    bottom: (box.bottom / maskHeight) * sourceHeight,
  };
}

function outputMaskSize(boxWidth: number, boxHeight: number): { width: number; height: number } {
  const scale = Math.min(1, SAM3_MAX_OUTPUT_MASK_SIDE / Math.max(boxWidth, boxHeight, 1));
  return {
    width: Math.max(1, Math.round(boxWidth * scale)),
    height: Math.max(1, Math.round(boxHeight * scale)),
  };
}

function applySigmoidInPlace(values: Float32Array): Float32Array {
  for (let index = 0; index < values.length; index += 1) {
    values[index] = sigmoid(values[index] ?? 0);
  }

  return values;
}

function probabilitiesToSegmentation(
  probabilities: Float32Array,
  maskWidth: number,
  maskHeight: number,
  box: Rect,
  label: string,
  confidence: number,
  pixelThreshold: number,
): Segmentation {
  const packed = packBinaryMask(probabilities, maskWidth, maskHeight, pixelThreshold);
  const local: Segmentation = new Segmentation({
    label: { index: 0, name: label },
    confidence,
    boundingBox: { left: 0, top: 0, right: maskWidth, bottom: maskHeight },
    bitPackedPixelMask: packed,
  });
  const localEdges = DrawTool.extractSegmentationEdgePoints(local);
  const scaleX = (box.right - box.left) / maskWidth;
  const scaleY = (box.bottom - box.top) / maskHeight;
  const edges: Point[] = localEdges.map(point => ({
    x: box.left + point.x * scaleX,
    y: box.top + point.y * scaleY,
  }));

  return new Segmentation({
    label: { index: 0, name: label },
    confidence,
    boundingBox: box,
    bitPackedPixelMask: packed,
    segmentationEdgePoints: edges,
    pixelMaskWidth: maskWidth,
    pixelMaskHeight: maskHeight,
  });
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
  const sourceBox = integerRect(
    boundingBox ??
      maskRectToSourceRect(
        maskBounds(logits, maskWidth, maskHeight, logitThreshold(pixelThreshold, applySigmoid)),
        maskWidth,
        maskHeight,
        sourceWidth,
        sourceHeight,
      ),
    sourceWidth,
    sourceHeight,
  );
  const boxWidth = sourceBox.right - sourceBox.left;
  const boxHeight = sourceBox.bottom - sourceBox.top;
  const output = outputMaskSize(boxWidth, boxHeight);
  const maskBox = sourceRectToMaskRect(sourceBox, sourceWidth, sourceHeight, maskWidth, maskHeight);
  const resized = bilinearResizeRegion(
    logits,
    maskWidth,
    maskHeight,
    maskBox.left,
    maskBox.top,
    maskBox.right,
    maskBox.bottom,
    output.width,
    output.height,
  );
  const probabilities = applySigmoid ? applySigmoidInPlace(resized) : resized;

  return probabilitiesToSegmentation(
    probabilities,
    output.width,
    output.height,
    sourceBox,
    label,
    confidence,
    pixelThreshold,
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
