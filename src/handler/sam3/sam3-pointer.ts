import type { Point, Rect } from '../../types';
import type { Sam3PointerLike, Sam3PointerToImageOptions } from './types';

export function pointerToImagePoint(
  event: Sam3PointerLike,
  target: HTMLElement | DOMRect,
  options: Sam3PointerToImageOptions = {},
): Point {
  const rect = isDomRect(target) ? target : target.getBoundingClientRect();
  const imageWidth =
    options.imageWidth ?? (target instanceof HTMLCanvasElement ? target.width : rect.width) ?? rect.width;
  const imageHeight =
    options.imageHeight ?? (target instanceof HTMLCanvasElement ? target.height : rect.height) ?? rect.height;
  const fit = options.objectFit ?? 'fill';

  if (fit === 'contain' || fit === 'cover') {
    const scale =
      fit === 'contain'
        ? Math.min(rect.width / Math.max(imageWidth, 1e-6), rect.height / Math.max(imageHeight, 1e-6))
        : Math.max(rect.width / Math.max(imageWidth, 1e-6), rect.height / Math.max(imageHeight, 1e-6));
    const drawWidth = imageWidth * scale;
    const drawHeight = imageHeight * scale;
    const offsetX = (rect.width - drawWidth) / 2;
    const offsetY = (rect.height - drawHeight) / 2;
    return {
      x: clamp((event.clientX - rect.left - offsetX) / Math.max(scale, 1e-6), 0, imageWidth),
      y: clamp((event.clientY - rect.top - offsetY) / Math.max(scale, 1e-6), 0, imageHeight),
    };
  }

  if (options.origin === 'offset' && event.offsetX != null && event.offsetY != null) {
    const element = target instanceof HTMLElement ? target : null;
    const cssWidth = element?.clientWidth || rect.width || 1;
    const cssHeight = element?.clientHeight || rect.height || 1;
    return {
      x: clamp((event.offsetX / cssWidth) * imageWidth, 0, imageWidth),
      y: clamp((event.offsetY / cssHeight) * imageHeight, 0, imageHeight),
    };
  }

  return {
    x: clamp(rect.width ? ((event.clientX - rect.left) / rect.width) * imageWidth : 0, 0, imageWidth),
    y: clamp(rect.height ? ((event.clientY - rect.top) / rect.height) * imageHeight : 0, 0, imageHeight),
  };
}

export function mapImagePoint(
  point: Point,
  from: { width: number; height: number },
  to: { width: number; height: number },
): Point {
  return {
    x: clamp(from.width ? (point.x * to.width) / from.width : point.x, 0, to.width),
    y: clamp(from.height ? (point.y * to.height) / from.height : point.y, 0, to.height),
  };
}

export function mapImageBox(
  box: Rect,
  from: { width: number; height: number },
  to: { width: number; height: number },
): Rect {
  const topLeft = mapImagePoint({ x: box.left, y: box.top }, from, to);
  const bottomRight = mapImagePoint({ x: box.right, y: box.bottom }, from, to);
  return {
    left: Math.min(topLeft.x, bottomRight.x),
    top: Math.min(topLeft.y, bottomRight.y),
    right: Math.max(topLeft.x, bottomRight.x),
    bottom: Math.max(topLeft.y, bottomRight.y),
  };
}

function isDomRect(value: HTMLElement | DOMRect): value is DOMRect {
  return typeof DOMRect !== 'undefined' && value instanceof DOMRect;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(max) || max <= min) {
    return Math.max(min, value);
  }
  return Math.min(Math.max(value, min), max);
}
