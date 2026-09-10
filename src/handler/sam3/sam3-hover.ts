import type { Point } from '../../types';
import type {
  Sam3HoverPointOptions,
  Sam3HoverPreviewOptions,
  Sam3HoverResult,
  Sam3PointerLike,
  Sam3PointerToImageOptions,
} from './types';
import { pointerToImagePoint } from './sam3-pointer';

interface Sam3HoverHost {
  hoverPoint(point: Point, options?: Sam3HoverPointOptions): Promise<Sam3HoverResult>;
  hoverFromPointer(
    event: Sam3PointerLike,
    target: HTMLElement | DOMRect,
    options?: Sam3HoverPointOptions & Sam3PointerToImageOptions,
  ): Promise<Sam3HoverResult>;
  hoverFromDisplayPoint(
    point: Point,
    displayWidth: number,
    displayHeight: number,
    options?: Sam3HoverPointOptions,
  ): Promise<Sam3HoverResult>;
}

/**
 * 鼠标悬停预览：只保留最新一点，推理中的中间点丢弃。
 */
export class Sam3HoverPreview {
  private queued: Point | null = null;
  private queuedKind: 'encoded' | 'display' = 'encoded';
  private queuedDisplay: { width: number; height: number } | null = null;
  private running = false;
  private lastPoint: Point | null = null;
  private lastResult: Sam3HoverResult | null = null;
  onResult: ((result: Sam3HoverResult | null) => void) | null;

  constructor(
    private readonly host: Sam3HoverHost,
    private readonly options: Sam3HoverPreviewOptions = {},
  ) {
    this.onResult = options.onResult ?? null;
  }

  get result(): Sam3HoverResult | null {
    return this.lastResult;
  }

  get mask() {
    return this.lastResult?.mask ?? null;
  }

  get busy(): boolean {
    return this.running;
  }

  async idle(): Promise<void> {
    while (this.running || this.queued) {
      await new Promise<void>(resolve => {
        requestAnimationFrame(() => resolve());
      });
    }
  }

  /** 已编码图像坐标系中的点 */
  queuePoint(point: Point): void {
    this.queued = point;
    this.queuedKind = 'encoded';
    this.queuedDisplay = null;
    this.kick();
  }

  /** 显示坐标系中的点（例如标注画布上的图像像素） */
  queueDisplayPoint(point: Point, displayWidth: number, displayHeight: number): void {
    this.queued = point;
    this.queuedKind = 'display';
    this.queuedDisplay = { width: displayWidth, height: displayHeight };
    this.kick();
  }

  /** Pointer / Mouse 事件，自动映射到图像像素再编码 */
  queuePointer(
    event: Sam3PointerLike,
    target: HTMLElement | DOMRect,
    pointer?: Sam3PointerToImageOptions,
  ): void {
    const size = pointerImageSize(target, pointer);
    const local = pointerToImagePoint(event, target, {
      ...pointer,
      imageWidth: size.width,
      imageHeight: size.height,
    });
    this.queueDisplayPoint(local, size.width, size.height);
  }

  /** offsetX / offsetY（元素 CSS 像素） */
  queueOffset(offsetX: number, offsetY: number, target: HTMLElement, pointer?: Sam3PointerToImageOptions): void {
    this.queuePointer({ clientX: 0, clientY: 0, offsetX, offsetY }, target, {
      ...pointer,
      origin: 'offset',
    });
  }

  clear(): void {
    this.queued = null;
    this.lastPoint = null;
    this.lastResult = null;
    this.onResult?.(null);
  }

  /** 编码图坐标是否足够接近最近一次悬停结果，可直接作为点击确认。 */
  canReusePoint(point: Point, maxDistance?: number): boolean {
    const limit = maxDistance ?? this.options.confirmMaxDistance ?? 8;
    const previous = this.lastResult?.promptPoint;
    return Boolean(
      this.lastResult?.mask &&
        previous &&
        Math.hypot(point.x - previous.x, point.y - previous.y) <= limit,
    );
  }

  /**
   * 点击确认：距离最近悬停点足够近时直接返回预览掩码，否则按同一套 hover 后处理再推理。
   */
  async confirmPoint(point: Point, maxDistance?: number): Promise<Sam3HoverResult> {
    await this.idle();
    if (this.canReusePoint(point, maxDistance) && this.lastResult) {
      return this.lastResult;
    }

    const result = await this.host.hoverPoint(point, this.options);
    this.lastPoint = point;
    this.lastResult = result;
    this.onResult?.(result);
    return result;
  }

  private kick(): void {
    if (!this.running) {
      void this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.running) {
      return;
    }

    this.running = true;
    try {
      while (this.queued) {
        const kind = this.queuedKind;
        const point = this.queued;
        const display = this.queuedDisplay;
        this.queued = null;

        if (point && this.lastPoint) {
          const minMove = this.options.minMove ?? 2;
          if (Math.hypot(point.x - this.lastPoint.x, point.y - this.lastPoint.y) < minMove) {
            continue;
          }
        }

        const result =
          kind === 'display' && point && display
            ? await this.host.hoverFromDisplayPoint(point, display.width, display.height, this.options)
            : await this.host.hoverPoint(point!, this.options);
        this.lastPoint = point;
        this.lastResult = result;
        this.onResult?.(result);
      }
    } finally {
      this.running = false;
      if (this.queued) {
        void this.flush();
      }
    }
  }
}

function pointerImageSize(
  target: HTMLElement | DOMRect,
  options?: Sam3PointerToImageOptions,
): { width: number; height: number } {
  if (options?.imageWidth && options.imageHeight) {
    return { width: options.imageWidth, height: options.imageHeight };
  }

  if (target instanceof HTMLCanvasElement) {
    return { width: target.width, height: target.height };
  }

  const rect =
    typeof DOMRect !== 'undefined' && target instanceof DOMRect
      ? target
      : (target as HTMLElement).getBoundingClientRect();
  return {
    width: options?.imageWidth ?? rect.width,
    height: options?.imageHeight ?? rect.height,
  };
}
