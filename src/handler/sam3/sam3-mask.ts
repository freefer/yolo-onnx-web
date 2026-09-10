import { DrawTool } from '../../draw-tool';
import { Segmentation } from '../../types';
import type { Point, Rect } from '../../types';
import type {
  Sam3HoverPick,
  Sam3HoverResult,
  Sam3HoverSelectOptions,
  Sam3ImagePixelMask,
  Sam3MaskRasterOptions,
  Sam3PvsResult,
} from './types';

export { maskToPolygon, maskToPolygons } from './sam3-contours';

export function packedMaskSize(mask: Segmentation): { width: number; height: number } {
  if (mask.pixelMaskWidth && mask.pixelMaskHeight) {
    return {
      width: Math.max(1, Math.round(mask.pixelMaskWidth)),
      height: Math.max(1, Math.round(mask.pixelMaskHeight)),
    };
  }

  const width = Math.max(1, Math.round(mask.boundingBox.right - mask.boundingBox.left));
  const height = Math.max(1, Math.round(mask.boundingBox.bottom - mask.boundingBox.top));
  return { width, height };
}

export function pickBestMask(
  masks: readonly Segmentation[],
  ious: readonly number[] = [],
  options: { pick?: Sam3HoverPick; point?: Point } = {},
): Segmentation | null {
  if (masks.length === 0) {
    return null;
  }

  const pick = options.pick ?? (options.point ? 'smallest' : 'iou');
  if (pick === 'first') {
    return masks[0] ?? null;
  }

  if (pick === 'smallest') {
    const containing = options.point ? masks.filter(mask => pointInBox(options.point!, mask.boundingBox)) : [];
    const pool = containing.length > 0 ? containing : [...masks];
    pool.sort((left, right) => maskArea(left) - maskArea(right));
    return pool[0] ?? null;
  }

  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < Math.max(ious.length, masks.length); index += 1) {
    const value = ious[index] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return masks[bestIndex] ?? masks[0] ?? null;
}

/**
 * 从 PVS 多候选中选出预览/点击应展示的那一块：默认点选 smallest + 连通域裁剪。
 */
export function selectVisualMask(
  result: Sam3PvsResult,
  options: Sam3HoverSelectOptions = {},
  promptBox?: Rect | null,
): Sam3HoverResult {
  const promptPoint = options.promptPoint;
  const picked = pickBestMask(result.masks, result.ious, {
    pick: options.pick ?? (promptPoint ? 'smallest' : 'iou'),
    point: promptPoint,
  });
  const maskIndex = picked ? Math.max(0, result.masks.indexOf(picked)) : 0;
  let mask = picked;
  const isolateAt =
    promptPoint ??
    (promptBox
      ? {
          x: (promptBox.left + promptBox.right) / 2,
          y: (promptBox.top + promptBox.bottom) / 2,
        }
      : undefined);
  if (mask && options.isolateComponent !== false && isolateAt) {
    mask = isolateMaskComponent(mask, isolateAt);
  }

  return {
    ...result,
    mask,
    maskIndex,
    promptPoint,
    promptBox: promptBox ?? undefined,
  };
}

export function isolateMaskComponent(mask: Segmentation, point: Point): Segmentation {
  const size = packedMaskSize(mask);
  const width = size.width;
  const height = size.height;
  const box = mask.boundingBox;
  const boxWidth = Math.max(box.right - box.left, 1e-6);
  const boxHeight = Math.max(box.bottom - box.top, 1e-6);
  let startX = Math.round(((point.x - box.left) / boxWidth) * (width - 1));
  let startY = Math.round(((point.y - box.top) / boxHeight) * (height - 1));
  startX = Math.min(Math.max(0, startX), width - 1);
  startY = Math.min(Math.max(0, startY), height - 1);

  if (!isPackedMaskSet(mask.bitPackedPixelMask, startY * width + startX)) {
    const nearest = findNearestSetPixel(mask.bitPackedPixelMask, width, height, startX, startY, 64);
    if (!nearest) {
      return mask;
    }
    startX = nearest.x;
    startY = nearest.y;
  }

  const component = floodFillMask(mask.bitPackedPixelMask, width, height, startX, startY);
  if (component.count < 12) {
    return mask;
  }

  const cropWidth = component.maxX - component.minX + 1;
  const cropHeight = component.maxY - component.minY + 1;
  const cropped = new Uint8Array(Math.ceil((cropWidth * cropHeight) / 8));
  for (let y = component.minY; y <= component.maxY; y += 1) {
    for (let x = component.minX; x <= component.maxX; x += 1) {
      if (!component.visited[y * width + x]) {
        continue;
      }
      const local = (y - component.minY) * cropWidth + (x - component.minX);
      cropped[local >> 3] |= 1 << (local & 0b0111);
    }
  }

  const isolated = new Segmentation({
    label: mask.label,
    confidence: mask.confidence,
    boundingBox: {
      left: box.left + (component.minX / width) * boxWidth,
      top: box.top + (component.minY / height) * boxHeight,
      right: box.left + ((component.maxX + 1) / width) * boxWidth,
      bottom: box.top + ((component.maxY + 1) / height) * boxHeight,
    },
    bitPackedPixelMask: cropped,
    pixelMaskWidth: cropWidth,
    pixelMaskHeight: cropHeight,
  });
  isolated.segmentationEdgePoints = DrawTool.extractSegmentationEdgePoints(isolated);
  return isolated;
}

/** 把编码图上的 Segmentation 采样到显示/标注图像坐标系。 */
export function maskToImagePixels(mask: Segmentation, options: Sam3MaskRasterOptions): Sam3ImagePixelMask | null {
  const imageWidth = options.imageWidth;
  const imageHeight = options.imageHeight;
  const sourceWidth = options.sourceWidth || imageWidth;
  const sourceHeight = options.sourceHeight || imageHeight;
  const size = packedMaskSize(mask);
  const packed = mask.bitPackedPixelMask;
  if (imageWidth <= 0 || imageHeight <= 0 || packed.byteLength * 8 < size.width * size.height) {
    return null;
  }

  const displayBox = {
    left: (mask.boundingBox.left * imageWidth) / sourceWidth,
    top: (mask.boundingBox.top * imageHeight) / sourceHeight,
    right: (mask.boundingBox.right * imageWidth) / sourceWidth,
    bottom: (mask.boundingBox.bottom * imageHeight) / sourceHeight,
  };
  const left = Math.max(0, Math.floor(displayBox.left));
  const top = Math.max(0, Math.floor(displayBox.top));
  const right = Math.min(imageWidth, Math.ceil(displayBox.right));
  const bottom = Math.min(imageHeight, Math.ceil(displayBox.bottom));
  const width = right - left;
  const height = bottom - top;
  if (width <= 0 || height <= 0) {
    return null;
  }

  const sampled = new Uint8Array(width * height);
  let area = 0;
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(
      size.height - 1,
      Math.max(
        0,
        Math.floor(
          ((top + y + 0.5 - displayBox.top) / Math.max(displayBox.bottom - displayBox.top, 1e-6)) * size.height,
        ),
      ),
    );
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(
        size.width - 1,
        Math.max(
          0,
          Math.floor(
            ((left + x + 0.5 - displayBox.left) / Math.max(displayBox.right - displayBox.left, 1e-6)) * size.width,
          ),
        ),
      );
      if (!isPackedMaskSet(packed, sourceY * size.width + sourceX)) {
        continue;
      }
      sampled[y * width + x] = 1;
      area += 1;
    }
  }

  return cropBinaryPixels(left, top, width, height, sampled, area);
}

function cropBinaryPixels(
  originX: number,
  originY: number,
  width: number,
  height: number,
  pixels: Uint8Array,
  area: number,
): Sam3ImagePixelMask | null {
  if (area <= 0) {
    return null;
  }

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!pixels[y * width + x]) {
        continue;
      }
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) {
    return null;
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const cropped = new Uint8Array(cropWidth * cropHeight);
  const packed = new Uint8Array(Math.ceil((cropWidth * cropHeight) / 8));
  let cropArea = 0;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (!pixels[y * width + x]) {
        continue;
      }
      const local = (y - minY) * cropWidth + (x - minX);
      cropped[local] = 1;
      packed[local >> 3] |= 1 << (local & 0b0111);
      cropArea += 1;
    }
  }

  return {
    x: originX + minX,
    y: originY + minY,
    width: cropWidth,
    height: cropHeight,
    area: cropArea,
    pixels: cropped,
    packed,
  };
}

function floodFillMask(
  packed: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
): { visited: Uint8Array; minX: number; maxX: number; minY: number; maxY: number; count: number } {
  const visited = new Uint8Array(width * height);
  const queue = [startY * width + startX];
  visited[startY * width + startX] = 1;
  let head = 0;
  let minX = startX;
  let maxX = startX;
  let minY = startY;
  let maxY = startY;
  let count = 0;

  while (head < queue.length) {
    const index = queue[head];
    head += 1;
    count += 1;
    const x = index % width;
    const y = (index / width) | 0;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    const neighbors = [index - 1, index + 1, index - width, index + width];
    const valid = [x > 0, x + 1 < width, y > 0, y + 1 < height];
    for (let offset = 0; offset < neighbors.length; offset += 1) {
      if (!valid[offset]) {
        continue;
      }
      const next = neighbors[offset];
      if (visited[next] || !isPackedMaskSet(packed, next)) {
        continue;
      }
      visited[next] = 1;
      queue.push(next);
    }
  }

  return { visited, minX, maxX, minY, maxY, count };
}

function findNearestSetPixel(
  packed: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  maxRadius: number,
): Point | null {
  let best: Point | null = null;
  let bestDistance = maxRadius * maxRadius;
  const minX = Math.max(0, startX - maxRadius);
  const maxX = Math.min(width - 1, startX + maxRadius);
  const minY = Math.max(0, startY - maxRadius);
  const maxY = Math.min(height - 1, startY + maxRadius);
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (!isPackedMaskSet(packed, y * width + x)) {
        continue;
      }
      const distance = (x - startX) * (x - startX) + (y - startY) * (y - startY);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = { x, y };
      }
    }
  }
  return best;
}

function pointInBox(point: Point, box: Rect): boolean {
  return point.x >= box.left && point.x <= box.right && point.y >= box.top && point.y <= box.bottom;
}

function maskArea(mask: Segmentation): number {
  return (
    Math.max(0, mask.boundingBox.right - mask.boundingBox.left) *
    Math.max(0, mask.boundingBox.bottom - mask.boundingBox.top)
  );
}

function isPackedMaskSet(packed: Uint8Array, pixelIndex: number): boolean {
  if (pixelIndex < 0) {
    return false;
  }

  const byteIndex = pixelIndex >> 3;
  if (byteIndex >= packed.byteLength) {
    return false;
  }

  return (packed[byteIndex] & (1 << (pixelIndex & 0b0111))) !== 0;
}
