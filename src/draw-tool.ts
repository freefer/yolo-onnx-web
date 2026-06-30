import type {
  Classification,
  ClassificationDrawingOptions,
  DetectionDrawingOptions,
  KeyPoint,
  OBBDetection,
  ObjectDetection,
  PoseDrawingOptions,
  PoseEstimation,
  Point,
  Rect,
  Segmentation,
  SegmentationDrawingOptions,
  YoloImageSource,
} from './types';

const DEFAULT_BOX_COLORS = [
  '#22c55e',
  '#3b82f6',
  '#f97316',
  '#e11d48',
  '#8b5cf6',
  '#14b8a6',
  '#f59e0b',
  '#06b6d4',
] as const;

const DEFAULT_POSE_CONNECTIONS: readonly [number, number][] = [
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
];

const EDGE_NEIGHBOR_OFFSETS = [
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
  { x: -1, y: 1 },
  { x: -1, y: 0 },
  { x: -1, y: -1 },
  { x: 0, y: -1 },
  { x: 1, y: -1 },
] as const;
const DEFAULT_EDGE_FILL_OPACITY = 64;

export class DrawTool {
  static drawObjectDetections(
    source: YoloImageSource,
    detections: readonly ObjectDetection[],
    canvas: HTMLCanvasElement,
    options: DetectionDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    this.drawBoundingBoxes(context, detections, width, height, options);
  }

  static drawClassifications(
    source: YoloImageSource,
    classifications: readonly Classification[],
    canvas: HTMLCanvasElement,
    options: ClassificationDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const fontColor = options.fontColor ?? '#f8fafc';
    const backgroundColor = options.backgroundColor ?? 'rgba(15, 23, 42, 0.72)';
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;
    const margin = 10;
    const lineGap = 8;

    context.font = font;
    context.textBaseline = 'top';

    const lineHeight = this.getCanvasFontSize(font) + lineGap;
    const labels = classifications.map(item => `${item.label}${drawConfidenceScore ? ` (${(item.confidence * 100).toFixed(1)}%)` : ''}`);
    const boxWidth = Math.max(0, ...labels.map(label => context.measureText(label).width)) + margin * 2;
    const boxHeight = labels.length * lineHeight + margin * 2 - lineGap;

    if (drawLabelBackground && labels.length > 0) {
      context.fillStyle = backgroundColor;
      context.fillRect(8, 8, boxWidth, boxHeight);
    }

    context.fillStyle = fontColor;

    for (let i = 0; i < labels.length; i += 1) {
      context.fillText(labels[i], 8 + margin, 8 + margin + i * lineHeight);
    }
  }

  static drawObbDetections(
    source: YoloImageSource,
    detections: readonly OBBDetection[],
    canvas: HTMLCanvasElement,
    options: DetectionDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 320));
    const drawLabel = options.drawLabel ?? true;
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const alpha = this.getDetectionDrawingAlpha(options);
    const fontColor = this.withAlpha(options.fontColor ?? '#f8fafc', alpha);

    context.font = font;
    context.textBaseline = 'middle';
    context.lineWidth = lineWidth;

    for (const detection of detections) {
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, alpha);
      const points = this.getObbCorners(detection.boundingBox, detection.orientationAngle);

      context.strokeStyle = color;
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        context.lineTo(points[i].x, points[i].y);
      }
      context.closePath();
      context.stroke();

      if (drawLabel) {
        this.drawDetectionLabel(context, detection, points[2].x, points[2].y, color, {
          font,
          drawConfidenceScore,
          drawLabelBackground,
          fontColor,
        });
      }
    }
  }

  static drawSegmentations(
    source: YoloImageSource,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const drawMask = options.drawSegmentationPixelMask ?? true;
    const drawContour = options.drawContour ?? false;
    const drawBoundingBoxes = options.drawBoundingBoxes ?? true;
    const pixelMaskOpacity = options.pixelMaskOpacity ?? 128;

    if (drawMask) {
      for (const segmentation of segmentations) {
        this.drawSegmentationMask(context, segmentation, this.getDetectionColor(segmentation, colors, undefined, pixelMaskOpacity));
      }
    }

    if (drawContour) {
      for (let index = 0; index < segmentations.length; index += 1) {
        const segmentation = segmentations[index];

        this.drawSegmentationContour(
          context,
          segmentation,
          this.getDetectionColor(segmentation, colors, options.strokeStyle),
          options.contourThickness ?? 2,

        );
      }
    }

    if (drawBoundingBoxes || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, segmentations, width, height, options);
    }
  }

  static drawPoseEstimations(
    source: YoloImageSource,
    poseEstimations: readonly PoseEstimation[],
    canvas: HTMLCanvasElement,
    options: PoseDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const confidence = options.poseConfidence ?? 0.25;
    const defaultPoseColor = options.defaultPoseColor ?? '#22c55e';
    const radius = options.keyPointRadius ?? Math.max(3, Math.round(Math.min(width, height) / 260));
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 360));
    const markers = options.keyPointMarkers;

    context.lineWidth = lineWidth;

    for (const pose of poseEstimations) {
      this.drawPoseConnections(context, pose.keyPoints, confidence, markers, defaultPoseColor);

      for (let i = 0; i < pose.keyPoints.length; i += 1) {
        const keyPoint = pose.keyPoints[i];

        if (keyPoint.confidence < confidence) {
          continue;
        }

        context.fillStyle = markers?.[i]?.color ?? defaultPoseColor;
        context.beginPath();
        context.arc(keyPoint.x, keyPoint.y, radius, 0, Math.PI * 2);
        context.fill();
      }
    }

    if ((options.drawBoundingBoxes ?? true) || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, poseEstimations, width, height, options);
    }
  }

  static extractSegmentationEdgePoints(segmentation: Segmentation): { x: number; y: number }[] {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return [];
    }

    const edgeKeys = new Set<number>();

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixelIndex = y * width + x;

        if (!this.isSegmentationEdgePixel(segmentation.bitPackedPixelMask, pixelIndex, x, y, width, height)) {
          continue;
        }

        edgeKeys.add(pixelIndex);
      }
    }

    return this.traceOrderedEdgePoints(edgeKeys, left, top, width);
  }

  static extractSegmentationsEdgePoints(segmentations: readonly Segmentation[]): { x: number; y: number }[][] {
    return segmentations.map(segmentation => this.extractSegmentationEdgePoints(segmentation));
  }

  private static traceOrderedEdgePoints(edgeKeys: Set<number>, left: number, top: number, width: number): { x: number; y: number }[] {
    const remaining = new Set(edgeKeys);
    const ordered: { x: number; y: number }[] = [];

    while (remaining.size > 0) {
      const startKey = this.getTopLeftKey(remaining, width);
      const contour = this.traceEdgeComponent(startKey, remaining, width);

      for (const key of contour) {
        ordered.push({
          x: left + key % width,
          y: top + Math.floor(key / width),
        });
      }
    }

    return ordered;
  }

  private static traceEdgeComponent(startKey: number, remaining: Set<number>, width: number): number[] {
    const contour: number[] = [];
    let currentKey = startKey;
    let previousKey = -1;
    let directionIndex = 0;

    while (remaining.has(currentKey)) {
      contour.push(currentKey);
      remaining.delete(currentKey);

      const next = this.getNextEdgeNeighbor(currentKey, previousKey, directionIndex, remaining, width);

      if (!next) {
        break;
      }

      previousKey = currentKey;
      currentKey = next.key;
      directionIndex = next.directionIndex;
    }

    return contour;
  }

  private static getNextEdgeNeighbor(
    currentKey: number,
    previousKey: number,
    directionIndex: number,
    remaining: Set<number>,
    width: number,
  ): { key: number; directionIndex: number } | null {
    const currentX = currentKey % width;
    const currentY = Math.floor(currentKey / width);
    const preferredDirection = previousKey >= 0
      ? this.getDirectionIndex(previousKey, currentKey, width)
      : directionIndex;

    for (let offset = -2; offset < EDGE_NEIGHBOR_OFFSETS.length - 2; offset += 1) {
      const candidateDirection = (preferredDirection + offset + EDGE_NEIGHBOR_OFFSETS.length) % EDGE_NEIGHBOR_OFFSETS.length;
      const neighbor = EDGE_NEIGHBOR_OFFSETS[candidateDirection];
      const nextX = currentX + neighbor.x;
      const nextY = currentY + neighbor.y;

      if (nextX < 0 || nextX >= width || nextY < 0) {
        continue;
      }

      const nextKey = nextY * width + nextX;

      if (remaining.has(nextKey)) {
        return { key: nextKey, directionIndex: candidateDirection };
      }
    }

    return null;
  }

  private static getDirectionIndex(fromKey: number, toKey: number, width: number): number {
    const fromX = fromKey % width;
    const fromY = Math.floor(fromKey / width);
    const toX = toKey % width;
    const toY = Math.floor(toKey / width);
    const dx = Math.sign(toX - fromX);
    const dy = Math.sign(toY - fromY);
    const index = EDGE_NEIGHBOR_OFFSETS.findIndex(offset => offset.x === dx && offset.y === dy);

    return index >= 0 ? index : 0;
  }

  private static getTopLeftKey(keys: Set<number>, width: number): number {
    let topLeftKey = -1;
    let topLeftX = Number.POSITIVE_INFINITY;
    let topLeftY = Number.POSITIVE_INFINITY;

    for (const key of keys) {
      const x = key % width;
      const y = Math.floor(key / width);

      if (y < topLeftY || (y === topLeftY && x < topLeftX)) {
        topLeftKey = key;
        topLeftX = x;
        topLeftY = y;
      }
    }

    return topLeftKey;
  }

  private static drawBoundingBoxes(
    context: CanvasRenderingContext2D,
    detections: readonly ObjectDetection[],
    width: number,
    height: number,
    options: DetectionDrawingOptions = {},
  ): void {
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const lineWidth = options.lineWidth ?? Math.max(2, Math.round(Math.min(width, height) / 320));
    const font = options.font ?? `${Math.max(14, Math.round(Math.min(width, height) / 70))}px Arial`;
    const drawLabel = options.drawLabel ?? true;
    const drawConfidenceScore = options.drawConfidenceScore ?? true;
    const drawLabelBackground = options.drawLabelBackground ?? true;
    const alpha = this.getDetectionDrawingAlpha(options);
    const fontColor = this.withAlpha(options.fontColor ?? '#f8fafc', alpha);

    context.lineWidth = lineWidth;
    context.font = font;
    context.textBaseline = 'middle';

    for (const detection of detections) {
      const { left, top, right, bottom } = detection.boundingBox;
      const boxWidth = right - left;
      const boxHeight = bottom - top;
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, alpha);

      if (boxWidth <= 0 || boxHeight <= 0) {
        continue;
      }

      context.strokeStyle = color;
      context.strokeRect(left, top, boxWidth, boxHeight);

      if (drawLabel) {
        this.drawDetectionLabel(context, detection, left, Math.max(0, top - this.getCanvasFontSize(font)), color, {
          font,
          drawConfidenceScore,
          drawLabelBackground,
          fontColor,
        });
      }
    }
  }

  private static prepareDrawingCanvas(
    source: YoloImageSource,
    canvas: HTMLCanvasElement,
    drawSource = true,
  ): { context: CanvasRenderingContext2D; width: number; height: number } {
    const { width, height } = this.getImageSourceSize(source);
    const context = canvas.getContext('2d');

    if (!context) {
      throw new Error('Canvas 2D context is not available.');
    }

    if (canvas.width !== width) {
      canvas.width = width;
    }

    if (canvas.height !== height) {
      canvas.height = height;
    }

    context.clearRect(0, 0, width, height);

    if (drawSource) {
      context.drawImage(source, 0, 0, width, height);
    }

    return { context, width, height };
  }

  private static getDetectionColor(
    detection: ObjectDetection,
    colors: readonly string[],
    fallback?: string,
    alpha = 255,
  ): string {
    const color = fallback ?? colors[detection.label.index % colors.length] ?? DEFAULT_BOX_COLORS[0];
    return this.withAlpha(color, alpha);
  }

  private static getDetectionDrawingAlpha(options: DetectionDrawingOptions): number {
    if (options.resultOpacity !== undefined) {
      return Math.round(this.clamp(options.resultOpacity, 0, 1) * 255);
    }

    return options.boundingBoxOpacity ?? 255;
  }

  private static withAlpha(color: string, alpha: number): string {
    if (!color.startsWith('#') || color.length !== 7) {
      return color;
    }

    const r = Number.parseInt(color.slice(1, 3), 16);
    const g = Number.parseInt(color.slice(3, 5), 16);
    const b = Number.parseInt(color.slice(5, 7), 16);
    const normalizedAlpha = this.clamp(alpha, 0, 255) / 255;

    return `rgba(${r}, ${g}, ${b}, ${normalizedAlpha})`;
  }

  private static drawDetectionLabel(
    context: CanvasRenderingContext2D,
    detection: ObjectDetection,
    x: number,
    y: number,
    backgroundColor: string,
    options: {
      font: string;
      drawConfidenceScore: boolean;
      drawLabelBackground: boolean;
      fontColor?: string;
    },
  ): void {
    const fontSize = this.getCanvasFontSize(options.font);
    const margin = Math.max(4, Math.round(fontSize / 3));
    const label = `${detection.label.name}${options.drawConfidenceScore ? ` ${(detection.confidence * 100).toFixed(1)}%` : ''}`;
    const textWidth = context.measureText(label).width;
    const boxWidth = textWidth + margin * 2;
    const boxHeight = fontSize + margin * 2;
    const left = this.clamp(Math.round(x), 0, Math.max(0, context.canvas.width - boxWidth));
    const top = this.clamp(Math.round(y), 0, Math.max(0, context.canvas.height - boxHeight));

    context.font = options.font;
    context.textBaseline = 'middle';

    if (options.drawLabelBackground) {
      context.fillStyle = backgroundColor;
      context.fillRect(left, top, boxWidth, boxHeight);
    }

    context.fillStyle = options.fontColor ?? '#f8fafc';
    context.fillText(label, left + margin, top + boxHeight / 2);
  }

  private static getCanvasFontSize(font: string): number {
    const match = font.match(/(\d+(?:\.\d+)?)px/);
    return match ? Number(match[1]) : 14;
  }

  private static getObbCorners(box: Rect, radians: number): { x: number; y: number }[] {
    const centerX = (box.left + box.right) / 2;
    const centerY = (box.top + box.bottom) / 2;
    const cos = Math.cos(radians);
    const sin = Math.sin(radians);
    const corners = [
      { x: box.left, y: box.top },
      { x: box.right, y: box.top },
      { x: box.right, y: box.bottom },
      { x: box.left, y: box.bottom },
    ];

    return corners.map(point => {
      const dx = point.x - centerX;
      const dy = point.y - centerY;

      return {
        x: centerX + dx * cos - dy * sin,
        y: centerY + dx * sin + dy * cos,
      };
    });
  }

  private static drawSegmentationMask(context: CanvasRenderingContext2D, segmentation: Segmentation, color: string): void {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }

    const imageData = context.createImageData(width, height);
    const rgba = this.parseCanvasColor(color);
    const maskCanvas = document.createElement('canvas');
    const maskContext = maskCanvas.getContext('2d');

    if (!maskContext) {
      throw new Error('Canvas 2D context is not available.');
    }

    maskCanvas.width = width;
    maskCanvas.height = height;

    for (let pixelIndex = 0; pixelIndex < width * height; pixelIndex += 1) {
      if (!this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex)) {
        continue;
      }

      const offset = pixelIndex * 4;
      imageData.data[offset] = rgba.r;
      imageData.data[offset + 1] = rgba.g;
      imageData.data[offset + 2] = rgba.b;
      imageData.data[offset + 3] = rgba.a;
    }

    maskContext.putImageData(imageData, 0, 0);
    context.drawImage(maskCanvas, left, top);
  }

  private static drawSegmentationContour(
    context: CanvasRenderingContext2D,
    segmentation: Segmentation,
    color: string,
    thickness: number,
  ): void {

    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }

    context.fillStyle = color;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixelIndex = y * width + x;

        if (!this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex)) {
          continue;
        }

        if (this.isSegmentationEdgePixel(segmentation.bitPackedPixelMask, pixelIndex, x, y, width, height)) {
          context.fillRect(left + x, top + y, thickness, thickness);
        }
      }
    }
  }

  public static drawSegmentationEdgePoints(
    source: YoloImageSource,
    segmentations: readonly Segmentation[],
    canvas: HTMLCanvasElement,
    options: SegmentationDrawingOptions = {},
  ): void {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const colors = options.boundingBoxHexColors ?? [...DEFAULT_BOX_COLORS];
    const thickness = options.contourThickness ?? 2;
    const drawBoundingBoxes = options.drawBoundingBoxes ?? true;

    for (let index = 0; index < segmentations.length; index += 1) {
      const segmentation = segmentations[index];
      const points = options.segmentationEdgePoints?.[index] ?? segmentation.segmentationEdgePoints ?? this.extractSegmentationEdgePoints(segmentation);
      const strokeColor = this.getDetectionColor(segmentation, colors, options.strokeStyle);
      const fillColor = this.getDetectionColor(
        segmentation,
        colors,
        options.fillStyle,
        options.pixelMaskOpacity ?? DEFAULT_EDGE_FILL_OPACITY,
      );
      if (options.drawSegmentationPixelMask === true) {
        this.drawOrderedEdgePoints(
          context,
          points,
          strokeColor,
          thickness,
          options.fillSegmentationEdgePoints === true ? fillColor : undefined,
        );
      }
    }

    if (drawBoundingBoxes || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, segmentations, width, height, options);
    }
  }

  private static drawOrderedEdgePoints(
    context: CanvasRenderingContext2D,
    points: readonly Point[],
    strokeColor: string,
    thickness: number,
    fillColor?: string,
  ): void {
    if (points.length === 0) {
      return;
    }

    context.save();
    context.lineWidth = thickness;
    context.lineJoin = 'round';
    context.lineCap = 'round';

    let previousPoint: Point | null = null;
    let hasPath = false;

    context.beginPath();

    for (const point of points) {
      if (!previousPoint || Math.abs(point.x - previousPoint.x) > 1 || Math.abs(point.y - previousPoint.y) > 1) {
        context.moveTo(point.x, point.y);
      } else {
        context.lineTo(point.x, point.y);
      }

      hasPath = true;
      previousPoint = point;
    }

    if (hasPath && fillColor) {
      context.closePath();
      context.fillStyle = fillColor;
      context.fill();
    }

    if (hasPath) {
      context.strokeStyle = strokeColor;
      context.stroke();
    }

    context.restore();
  }

  private static isSegmentationEdgePixel(
    mask: Uint8Array,
    pixelIndex: number,
    x: number,
    y: number,
    width: number,
    height: number,
  ): boolean {
    if (!this.isPackedMaskSet(mask, pixelIndex)) {
      return false;
    }

    return (
      x === 0 ||
      x === width - 1 ||
      y === 0 ||
      y === height - 1 ||
      !this.isPackedMaskSet(mask, pixelIndex - 1) ||
      !this.isPackedMaskSet(mask, pixelIndex + 1) ||
      !this.isPackedMaskSet(mask, pixelIndex - width) ||
      !this.isPackedMaskSet(mask, pixelIndex + width)
    );
  }

  private static isPackedMaskSet(mask: Uint8Array, pixelIndex: number): boolean {
    if (pixelIndex < 0) {
      return false;
    }

    const byteIndex = pixelIndex >> 3;

    if (byteIndex >= mask.byteLength) {
      return false;
    }

    return (mask[byteIndex] & (1 << (pixelIndex & 0b0111))) !== 0;
  }

  private static parseCanvasColor(color: string): { r: number; g: number; b: number; a: number } {
    if (color.startsWith('#') && color.length === 7) {
      return {
        r: Number.parseInt(color.slice(1, 3), 16),
        g: Number.parseInt(color.slice(3, 5), 16),
        b: Number.parseInt(color.slice(5, 7), 16),
        a: 255,
      };
    }

    const rgba = color.match(/rgba?\(([^)]+)\)/);

    if (rgba) {
      const parts = rgba[1].split(',').map(part => Number(part.trim()));

      return {
        r: parts[0] ?? 34,
        g: parts[1] ?? 197,
        b: parts[2] ?? 94,
        a: Math.round((parts[3] ?? 1) * 255),
      };
    }

    return { r: 34, g: 197, b: 94, a: 128 };
  }

  private static drawPoseConnections(
    context: CanvasRenderingContext2D,
    keyPoints: readonly KeyPoint[],
    confidence: number,
    markers: PoseDrawingOptions['keyPointMarkers'],
    defaultColor: string,
  ): void {
    if (markers && markers.length > 0) {
      for (let sourceIndex = 0; sourceIndex < markers.length; sourceIndex += 1) {
        const source = keyPoints[sourceIndex];

        if (!source || source.confidence < confidence) {
          continue;
        }

        for (const connection of markers[sourceIndex]?.connections ?? []) {
          this.drawPoseConnection(context, source, keyPoints[connection.index], confidence, connection.color ?? defaultColor);
        }
      }

      return;
    }

    for (const [sourceIndex, targetIndex] of DEFAULT_POSE_CONNECTIONS) {
      this.drawPoseConnection(context, keyPoints[sourceIndex], keyPoints[targetIndex], confidence, defaultColor);
    }
  }

  private static drawPoseConnection(
    context: CanvasRenderingContext2D,
    source: KeyPoint | undefined,
    target: KeyPoint | undefined,
    confidence: number,
    color: string,
  ): void {
    if (!source || !target || source.confidence < confidence || target.confidence < confidence) {
      return;
    }

    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(source.x, source.y);
    context.lineTo(target.x, target.y);
    context.stroke();
  }

  private static getImageSourceSize(img: YoloImageSource): { width: number; height: number } {
    if (img instanceof HTMLImageElement) {
      return {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
      };
    }

    if (img instanceof HTMLVideoElement) {
      return {
        width: img.videoWidth || img.width,
        height: img.videoHeight || img.height,
      };
    }

    if ('displayWidth' in img && 'displayHeight' in img) {
      return {
        width: img.displayWidth || img.codedWidth,
        height: img.displayHeight || img.codedHeight,
      };
    }

    if (img instanceof SVGImageElement) {
      const width = img.width.baseVal.value || img.getBoundingClientRect().width;
      const height = img.height.baseVal.value || img.getBoundingClientRect().height;

      return { width, height };
    }

    return {
      width: img.width,
      height: img.height,
    };
  }

  private static clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }
}
