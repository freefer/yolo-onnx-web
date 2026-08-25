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

  context.drawImage(image, 0, 0, sourceWidth, sourceHeight, 0, 0, imageSize, imageSize);
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
