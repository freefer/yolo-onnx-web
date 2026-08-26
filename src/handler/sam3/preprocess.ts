import type { Point, Rect, YoloImageSource } from '../../types';
import type { Sam3ImageTensor } from './types';

export const SAM3_IMAGE_SIZE = 1008;
export const SAM3_MASK_SIZE = 288;
export const SAM3_TEXT_LENGTH = 32;
export const SAM3_MEAN = 0.5;
export const SAM3_STD = 0.5;

export function getImageSize(image: YoloImageSource): { width: number; height: number } {
  if (image instanceof HTMLVideoElement) {
    return { width: image.videoWidth, height: image.videoHeight };
  }

  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth || image.width, height: image.naturalHeight || image.height };
  }

  if (image instanceof HTMLCanvasElement || image instanceof OffscreenCanvas) {
    return { width: image.width, height: image.height };
  }

  if ('displayWidth' in image && 'displayHeight' in image) {
    return { width: Number(image.displayWidth), height: Number(image.displayHeight) };
  }

  const sized = image as { width: number; height: number };
  return { width: sized.width, height: sized.height };
}

function enableHighQualitySmoothing(context: CanvasRenderingContext2D): void {
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
}

/**
 * Progressive downsample before the official 1008×1008 stretch.
 * A single canvas blit from a huge photo to 1008 aliases badly and tanks scores.
 */
export function drawImageHighQuality(
  context: CanvasRenderingContext2D,
  image: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  destWidth: number,
  destHeight: number,
): void {
  enableHighQualitySmoothing(context);

  let src: CanvasImageSource = image;
  let width = sourceWidth;
  let height = sourceHeight;

  while (width > destWidth * 2 || height > destHeight * 2) {
    const nextWidth = Math.max(destWidth, Math.floor(width / 2));
    const nextHeight = Math.max(destHeight, Math.floor(height / 2));
    const tmp = document.createElement('canvas');
    tmp.width = nextWidth;
    tmp.height = nextHeight;
    const tmpContext = tmp.getContext('2d');

    if (!tmpContext) {
      break;
    }

    enableHighQualitySmoothing(tmpContext);
    tmpContext.drawImage(src, 0, 0, width, height, 0, 0, nextWidth, nextHeight);
    src = tmp;
    width = nextWidth;
    height = nextHeight;
  }

  context.drawImage(src, 0, 0, width, height, 0, 0, destWidth, destHeight);
}

export function preprocessSam3Image(
  image: YoloImageSource,
  imageSize = SAM3_IMAGE_SIZE,
): Sam3ImageTensor {
  const { width: sourceWidth, height: sourceHeight } = getImageSize(image);
  const canvas = document.createElement('canvas');
  canvas.width = imageSize;
  canvas.height = imageSize;
  const context = canvas.getContext('2d', { willReadFrequently: true });

  if (!context) {
    throw new Error('Failed to create a 2D canvas context for SAM3 preprocessing.');
  }

  drawImageHighQuality(context, image, sourceWidth, sourceHeight, imageSize, imageSize);
  const pixels = context.getImageData(0, 0, imageSize, imageSize).data;
  const plane = imageSize * imageSize;
  const data = new Float32Array(3 * plane);

  for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
    data[pixel] = (pixels[offset] / 255 - SAM3_MEAN) / SAM3_STD;
    data[plane + pixel] = (pixels[offset + 1] / 255 - SAM3_MEAN) / SAM3_STD;
    data[plane * 2 + pixel] = (pixels[offset + 2] / 255 - SAM3_MEAN) / SAM3_STD;
  }

  return { data, width: imageSize, height: imageSize, sourceWidth, sourceHeight };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

export function splitTextPrompts(prompt: string): string[] {
  const trimmed = prompt.trim();

  if (!trimmed) {
    return [];
  }

  const parts = trimmed.split(/[,，]+/).map(part => part.trim()).filter(Boolean);
  return parts.length > 0 ? parts : [trimmed];
}

export function composeTextQuery(className: string, description?: string): string {
  const extra = description?.trim();
  return extra ? `${className}, ${extra}` : className;
}

/**
 * grounding-decoder.onnx was traced with 1 box + 1 point (+ 1 CLS inside the graph).
 * That freezes the fused prompt length at 32 (text) + 3 = 35. Feeding a second box
 * plus the dummy point becomes 36 and crashes ORT reshape / the WASM page.
 */
export interface PackedPcsBox {
  box: Rect;
  label: boolean;
  pad: boolean;
}

export interface PackedPcsPoint {
  point: Point;
  label: 0 | 1;
  pad: boolean;
}

export function rectCenter(box: Rect): Point {
  return { x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2 };
}

/**
 * Pack PCS exemplars into the 1-box + 1-point layout the current ONNX graph accepts.
 * Extra boxes become a point at the previous box center so two box draws still both count.
 */
export function packPcsGeometricPrompts(
  boxes: readonly { box: Rect; label: boolean }[],
  points: readonly { point: Point; label: number }[],
): { boxes: PackedPcsBox[]; points: PackedPcsPoint[] } {
  const lastBox = boxes[boxes.length - 1];
  const packedBox: PackedPcsBox = lastBox
    ? { box: lastBox.box, label: lastBox.label, pad: false }
    : { box: { left: 0, top: 0, right: 1, bottom: 1 }, label: true, pad: true };

  const lastPoint = points[points.length - 1];
  let packedPoint: PackedPcsPoint;

  if (lastPoint) {
    packedPoint = { point: lastPoint.point, label: lastPoint.label > 0 ? 1 : 0, pad: false };
  } else {
    const extraBox = boxes.length >= 2 ? boxes[boxes.length - 2] : undefined;
    packedPoint = extraBox
      ? { point: rectCenter(extraBox.box), label: extraBox.label ? 1 : 0, pad: false }
      : { point: { x: 0, y: 0 }, label: 1, pad: true };
  }

  return { boxes: [packedBox], points: [packedPoint] };
}

export function rectToCxcywh(box: Rect, sourceWidth: number, sourceHeight: number): [number, number, number, number] {
  const width = Math.max(box.right - box.left, 1);
  const height = Math.max(box.bottom - box.top, 1);
  return [
    clamp((box.left + box.right) / 2 / sourceWidth, 0, 1),
    clamp((box.top + box.bottom) / 2 / sourceHeight, 0, 1),
    clamp(width / sourceWidth, 0, 1),
    clamp(height / sourceHeight, 0, 1),
  ];
}

export function cxcywhToRect(
  cx: number,
  cy: number,
  width: number,
  height: number,
  sourceWidth: number,
  sourceHeight: number,
): Rect {
  const left = (cx - width / 2) * sourceWidth;
  const top = (cy - height / 2) * sourceHeight;
  const right = (cx + width / 2) * sourceWidth;
  const bottom = (cy + height / 2) * sourceHeight;

  return {
    left: clamp(left, 0, sourceWidth),
    top: clamp(top, 0, sourceHeight),
    right: clamp(right, 0, sourceWidth),
    bottom: clamp(bottom, 0, sourceHeight),
  };
}

export function pointToNormalized(point: Point, sourceWidth: number, sourceHeight: number): [number, number] {
  return [clamp(point.x / sourceWidth, 0, 1), clamp(point.y / sourceHeight, 0, 1)];
}

export function pointToModel(point: Point, sourceWidth: number, sourceHeight: number, imageSize = SAM3_IMAGE_SIZE): [number, number] {
  return [
    (point.x / sourceWidth) * imageSize,
    (point.y / sourceHeight) * imageSize,
  ];
}

export function boxToModelCorners(
  box: Rect,
  sourceWidth: number,
  sourceHeight: number,
  imageSize = SAM3_IMAGE_SIZE,
): Array<[number, number]> {
  return [
    pointToModel({ x: box.left, y: box.top }, sourceWidth, sourceHeight, imageSize),
    pointToModel({ x: box.right, y: box.bottom }, sourceWidth, sourceHeight, imageSize),
  ];
}
