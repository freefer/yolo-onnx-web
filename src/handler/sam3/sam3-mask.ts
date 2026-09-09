import { DrawTool } from '../../draw-tool';
import { Segmentation } from '../../types';
import type { Point, Rect } from '../../types';
import type { Sam3HoverPick, Sam3MaskPolygonOptions } from './types';

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

  return new Segmentation({
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
}

export function maskToPolygon(mask: Segmentation, options: Sam3MaskPolygonOptions): number[][] {
  const size = packedMaskSize(mask);
  const local = new Segmentation({
    label: mask.label,
    confidence: mask.confidence,
    boundingBox: { left: 0, top: 0, right: size.width, bottom: size.height },
    bitPackedPixelMask: mask.bitPackedPixelMask,
    pixelMaskWidth: size.width,
    pixelMaskHeight: size.height,
  });
  const sourceWidth = options.sourceWidth || options.imageWidth || 1;
  const sourceHeight = options.sourceHeight || options.imageHeight || 1;
  const scaleX = options.imageWidth / sourceWidth;
  const scaleY = options.imageHeight / sourceHeight;
  const box = mask.boundingBox;
  const boxWidth = Math.max(box.right - box.left, 1e-6);
  const boxHeight = Math.max(box.bottom - box.top, 1e-6);
  const contours = DrawTool.splitEdgeContours(DrawTool.extractSegmentationEdgePoints(local)).map(contour =>
    contour.map(point => ({
      x: (box.left + (point.x * boxWidth) / size.width) * scaleX,
      y: (box.top + (point.y * boxHeight) / size.height) * scaleY,
    })),
  );
  const prompt = options.prompt
    ? {
        x: options.prompt.x * scaleX,
        y: options.prompt.y * scaleY,
      }
    : undefined;
  const best = pickBestContour(contours, prompt);
  const raw = best.map(point => [
    clamp(point.x, 0, options.imageWidth),
    clamp(point.y, 0, options.imageHeight),
  ]);
  const simplified = simplifyRdp(raw, options.epsilon ?? 1.25);
  return simplified.length >= 3 ? simplified : raw;
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

function pickBestContour(contours: readonly Point[][], prompt?: Point): Point[] {
  if (contours.length === 0) {
    return [];
  }

  const ranked = [...contours].sort((left, right) => polygonArea(right) - polygonArea(left));
  const closed = ranked.filter(isMostlyClosedContour);
  const pool = closed.length > 0 ? closed : ranked;
  if (!prompt) {
    return pool[0] ?? [];
  }

  const containing = pool.filter(contour => pointInPolygon(prompt, contour));
  return containing[0] ?? pool[0] ?? [];
}

function isMostlyClosedContour(points: readonly Point[]): boolean {
  if (points.length < 3) {
    return false;
  }

  const first = points[0];
  const last = points[points.length - 1];
  const gap = Math.max(Math.abs(first.x - last.x), Math.abs(first.y - last.y));
  return gap <= Math.max(estimateEdgeStep(points) * 3, 4);
}

function estimateEdgeStep(points: readonly Point[]): number {
  return DrawTool.estimateEdgeStep(points);
}

function polygonArea(points: readonly Point[]): number {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current.x * next.y - next.x * current.y;
  }
  return Math.abs(area) / 2;
}

function pointInPolygon(point: Point, polygon: readonly Point[]): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
    const current = polygon[index];
    const last = polygon[previous];
    const intersects =
      current.y > point.y !== last.y > point.y &&
      point.x < ((last.x - current.x) * (point.y - current.y)) / ((last.y - current.y) || 1e-6) + current.x;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
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

function simplifyRdp(points: number[][], epsilon: number): number[][] {
  if (points.length <= 4) {
    return points;
  }

  const first = points[0];
  const last = points[points.length - 1];
  let maxDistance = 0;
  let maxIndex = 0;
  for (let index = 1; index < points.length - 1; index += 1) {
    const distance = perpendicularDistance(points[index], first, last);
    if (distance > maxDistance) {
      maxDistance = distance;
      maxIndex = index;
    }
  }

  if (maxDistance <= epsilon) {
    return [first, last];
  }

  const left = simplifyRdp(points.slice(0, maxIndex + 1), epsilon);
  const right = simplifyRdp(points.slice(maxIndex), epsilon);
  return [...left.slice(0, -1), ...right];
}

function perpendicularDistance(point: number[], start: number[], end: number[]): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const length = Math.hypot(dx, dy) || 1;
  return Math.abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) / length;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(max) || max <= min) {
    return Math.max(min, value);
  }
  return Math.min(Math.max(value, min), max);
}
