import {
  extractSegmentationPolygon,
  extractSegmentationPolygons,
} from '../../segmentation-contours';
import type { Segmentation } from '../../types';
import type { Sam3MaskPolygonOptions } from './types';

/** @deprecated Use extractSegmentationPolygons() instead. */
export function maskToPolygons(
  mask: Segmentation,
  options: Sam3MaskPolygonOptions,
): number[][][] {
  return extractSegmentationPolygons(mask, options).map(polygon =>
    polygon.map(point => [point.x, point.y]),
  );
}

/** @deprecated Use extractSegmentationPolygon() instead. */
export function maskToPolygon(
  mask: Segmentation,
  options: Sam3MaskPolygonOptions,
): number[][] {
  return extractSegmentationPolygon(mask, options).map(point => [point.x, point.y]);
}
