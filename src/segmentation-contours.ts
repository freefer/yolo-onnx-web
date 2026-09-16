import type { Point, Segmentation, SegmentationPolygonOptions } from './types';

const DEFAULT_MAX_POLYGON_POINTS = 96;
const MAX_MASK_POLYGONS = 16;
const MIN_MASK_POLYGON_AREA_RATIO = 0.01;

interface MaskBoundaryEdge {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  direction: number;
  used: boolean;
}

/**
 * 将任意模型返回的 bit-packed segmentation mask 转为图像坐标 polygon。
 * YOLO、RF-DETR 与 SAM3 共用该实现。
 */
export function extractSegmentationPolygons(
  segmentation: Segmentation,
  options: SegmentationPolygonOptions,
): Point[][] {
  return extractNumberPolygons(segmentation, options).map(polygon =>
    polygon.map(point => ({ x: point[0], y: point[1] })),
  );
}

export function extractSegmentationPolygon(
  segmentation: Segmentation,
  options: SegmentationPolygonOptions,
): Point[] {
  return extractSegmentationPolygons(segmentation, options)[0] ?? [];
}

function extractNumberPolygons(
  segmentation: Segmentation,
  options: SegmentationPolygonOptions,
): number[][][] {
  const imageWidth = options.imageWidth;
  const imageHeight = options.imageHeight;
  if (imageWidth <= 0 || imageHeight <= 0) {
    return [];
  }

  const contours = tracePackedMaskContours(segmentation);
  if (contours.length === 0) {
    return [];
  }

  const sourceWidth = options.sourceWidth || imageWidth || 1;
  const sourceHeight = options.sourceHeight || imageHeight || 1;
  const size = packedMaskSize(segmentation);
  const box = segmentation.boundingBox;
  const boxWidth = box.right - box.left;
  const boxHeight = box.bottom - box.top;
  const maxPoints = options.maxPoints ?? DEFAULT_MAX_POLYGON_POINTS;
  const mapped = contours
    .map(contour => {
      const points = contour.map(point => [
        clamp((box.left + (point[0] * boxWidth) / size.width) * (imageWidth / sourceWidth), 0, imageWidth),
        clamp((box.top + (point[1] * boxHeight) / size.height) * (imageHeight / sourceHeight), 0, imageHeight),
      ]);
      return { points, area: Math.abs(signedContourArea(points)) };
    })
    .filter(item => item.points.length >= 3 && item.area > 0)
    .sort((left, right) => right.area - left.area);

  if (mapped.length === 0) {
    return [];
  }

  if (options.prompt) {
    const mappedPrompt = {
      x: (options.prompt.x * imageWidth) / sourceWidth,
      y: (options.prompt.y * imageHeight) / sourceHeight,
    };
    const selected = mapped.find(item => pointInNumberContour(mappedPrompt, item.points)) ?? mapped[0];
    return [finalizePolygon(selected.points, imageWidth, imageHeight, maxPoints, options.epsilon)];
  }

  const minArea = mapped[0].area * MIN_MASK_POLYGON_AREA_RATIO;
  const outerContours: number[][][] = [];
  for (const candidate of mapped) {
    if (candidate.area < minArea || outerContours.length >= MAX_MASK_POLYGONS) {
      break;
    }
    const probe = { x: candidate.points[0][0], y: candidate.points[0][1] };
    if (outerContours.some(outer => pointInNumberContour(probe, outer))) {
      continue;
    }
    outerContours.push(finalizePolygon(candidate.points, imageWidth, imageHeight, maxPoints, options.epsilon));
  }

  return outerContours.filter(polygon => polygon.length >= 3);
}

function packedMaskSize(segmentation: Segmentation): { width: number; height: number } {
  if (segmentation.pixelMaskWidth && segmentation.pixelMaskHeight) {
    return {
      width: Math.max(1, Math.round(segmentation.pixelMaskWidth)),
      height: Math.max(1, Math.round(segmentation.pixelMaskHeight)),
    };
  }

  return {
    width: Math.max(1, Math.round(segmentation.boundingBox.right - segmentation.boundingBox.left)),
    height: Math.max(1, Math.round(segmentation.boundingBox.bottom - segmentation.boundingBox.top)),
  };
}

function tracePackedMaskContours(segmentation: Segmentation): number[][][] {
  const packed = segmentation.bitPackedPixelMask;
  const { width, height } = packedMaskSize(segmentation);
  if (!packed?.byteLength || width * height > packed.byteLength * 8) {
    return [];
  }

  const vertexWidth = width + 1;
  const edges: MaskBoundaryEdge[] = [];
  const outgoing = new Map<number, MaskBoundaryEdge[]>();
  const isSet = (x: number, y: number): boolean => {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return false;
    }
    const index = y * width + x;
    return (packed[index >> 3] & (1 << (index & 7))) !== 0;
  };
  const addEdge = (fromX: number, fromY: number, toX: number, toY: number, direction: number): void => {
    const edge: MaskBoundaryEdge = { fromX, fromY, toX, toY, direction, used: false };
    edges.push(edge);
    const key = fromY * vertexWidth + fromX;
    const next = outgoing.get(key);
    if (next) {
      next.push(edge);
    } else {
      outgoing.set(key, [edge]);
    }
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!isSet(x, y)) {
        continue;
      }
      if (!isSet(x, y - 1)) {
        addEdge(x, y, x + 1, y, 0);
      }
      if (!isSet(x + 1, y)) {
        addEdge(x + 1, y, x + 1, y + 1, 1);
      }
      if (!isSet(x, y + 1)) {
        addEdge(x + 1, y + 1, x, y + 1, 2);
      }
      if (!isSet(x - 1, y)) {
        addEdge(x, y + 1, x, y, 3);
      }
    }
  }

  const contours: number[][][] = [];
  for (const first of edges) {
    if (first.used) {
      continue;
    }
    const startKey = first.fromY * vertexWidth + first.fromX;
    const points: number[][] = [[first.fromX, first.fromY]];
    let current: MaskBoundaryEdge | undefined = first;
    let closed = false;

    for (let guard = 0; current && guard <= edges.length; guard += 1) {
      current.used = true;
      points.push([current.toX, current.toY]);
      const endKey = current.toY * vertexWidth + current.toX;
      if (endKey === startKey) {
        closed = true;
        break;
      }
      current = chooseNextBoundaryEdge(outgoing.get(endKey), current.direction);
    }

    if (!closed || points.length < 4) {
      continue;
    }
    points.pop();
    if (signedContourArea(points) > 0) {
      contours.push(points);
    }
  }
  return contours;
}

function chooseNextBoundaryEdge(
  candidates: readonly MaskBoundaryEdge[] | undefined,
  incomingDirection: number,
): MaskBoundaryEdge | undefined {
  let selected: MaskBoundaryEdge | undefined;
  let selectedRank = Number.POSITIVE_INFINITY;
  for (const candidate of candidates ?? []) {
    if (candidate.used) {
      continue;
    }
    const turn = (candidate.direction - incomingDirection + 4) % 4;
    const rank = turn === 1 ? 0 : turn === 0 ? 1 : turn === 3 ? 2 : 3;
    if (rank < selectedRank) {
      selected = candidate;
      selectedRank = rank;
    }
  }
  return selected;
}

function finalizePolygon(
  points: number[][],
  imageWidth: number,
  imageHeight: number,
  maxPoints: number,
  epsilon?: number,
): number[][] {
  const pointLimit = Math.min(512, Math.max(3, Math.round(maxPoints)));
  const dense = simplifyClosedPolygon(points, 0);
  if (dense.length <= pointLimit) {
    return dense.length >= 3 ? dense : samplePolygon(points, pointLimit);
  }

  let lowerEpsilon = 0;
  let upperEpsilon = epsilon ?? polygonEpsilon(imageWidth, imageHeight);
  let aboveLimit = dense;
  let belowLimit = simplifyClosedPolygon(points, upperEpsilon);
  while (belowLimit.length > pointLimit && upperEpsilon < 128) {
    lowerEpsilon = upperEpsilon;
    aboveLimit = belowLimit;
    upperEpsilon *= 2;
    belowLimit = simplifyClosedPolygon(points, upperEpsilon);
  }

  for (let iteration = 0; iteration < 16; iteration += 1) {
    const mid = (lowerEpsilon + upperEpsilon) / 2;
    const candidate = simplifyClosedPolygon(points, mid);
    if (candidate.length > pointLimit) {
      lowerEpsilon = mid;
      aboveLimit = candidate;
    } else {
      upperEpsilon = mid;
      belowLimit = candidate;
    }
  }

  if (aboveLimit.length > pointLimit) {
    return samplePolygon(aboveLimit, pointLimit);
  }
  return belowLimit.length >= 3 ? belowLimit : samplePolygon(dense, pointLimit);
}

function polygonEpsilon(width: number, height: number): number {
  return Math.max(1.25, Math.max(width, height) / 2048);
}

function signedContourArea(points: readonly number[][]): number {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current[0] * next[1] - next[0] * current[1];
  }
  return area / 2;
}

function pointInNumberContour(point: Point, contour: readonly number[][]): boolean {
  let inside = false;
  for (let index = 0, previous = contour.length - 1; index < contour.length; previous = index, index += 1) {
    const current = contour[index];
    const last = contour[previous];
    if (
      current[1] > point.y !== last[1] > point.y &&
      point.x < ((last[0] - current[0]) * (point.y - current[1])) / (last[1] - current[1] || 1e-6) + current[0]
    ) {
      inside = !inside;
    }
  }
  return inside;
}

function samplePolygon(points: number[][], limit: number): number[][] {
  if (points.length <= limit) {
    return points;
  }
  const sampled: number[][] = [];
  const step = points.length / limit;
  for (let index = 0; index < limit; index += 1) {
    sampled.push(points[Math.floor(index * step)]);
  }
  return sampled;
}

function simplifyClosedPolygon(points: number[][], epsilon: number): number[][] {
  if (points.length <= 4) {
    return points;
  }
  const closed = [...points, points[0]];
  const simplified = simplifyRdp(closed, epsilon);
  if (
    simplified.length > 1 &&
    simplified[0][0] === simplified[simplified.length - 1][0] &&
    simplified[0][1] === simplified[simplified.length - 1][1]
  ) {
    simplified.pop();
  }
  return simplified;
}

function simplifyRdp(points: number[][], epsilon: number): number[][] {
  if (points.length <= 4) {
    return points;
  }
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: Array<[number, number]> = [[0, points.length - 1]];

  while (stack.length > 0) {
    const [startIndex, endIndex] = stack.pop()!;
    let maxDistance = 0;
    let maxIndex = -1;
    for (let index = startIndex + 1; index < endIndex; index += 1) {
      const distance = perpendicularDistance(points[index], points[startIndex], points[endIndex]);
      if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = index;
      }
    }
    if (maxIndex >= 0 && maxDistance > epsilon) {
      keep[maxIndex] = 1;
      stack.push([startIndex, maxIndex], [maxIndex, endIndex]);
    }
  }

  return points.filter((_, index) => keep[index] === 1);
}

function perpendicularDistance(point: number[], start: number[], end: number[]): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const length = Math.hypot(dx, dy);
  if (length <= 1e-6) {
    return Math.hypot(point[0] - start[0], point[1] - start[1]);
  }
  return Math.abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) / length;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(max) || max <= min) {
    return Math.max(min, value);
  }
  return Math.min(Math.max(value, min), max);
}
