// src/runtime.ts
var ortModule = null;
var loadedBundle = null;
var loadingPromise = null;
function isWebAssemblyJspiAvailable() {
  const wasm = globalThis.WebAssembly;
  return Boolean(wasm && "Suspending" in wasm);
}
function resolveOrtBundle(executionProviders = ["wasm"], ortBundle = "auto") {
  if (ortBundle !== "auto") {
    return ortBundle;
  }
  const names = new Set(
    executionProviders.map((provider) => {
      var _a;
      if (typeof provider === "string") {
        return provider;
      }
      return (_a = provider.name) != null ? _a : "";
    })
  );
  if (names.has("webgpu")) {
    return isWebAssemblyJspiAvailable() ? "jspi" : "webgpu";
  }
  if (names.has("webgl")) {
    return "webgl";
  }
  if (names.has("webnn")) {
    return "all";
  }
  return "wasm";
}
function canReuseOrtBundle(loaded, requested) {
  if (loaded === requested) {
    return true;
  }
  if ((loaded === "webgpu" || loaded === "jspi") && requested === "wasm") {
    return true;
  }
  if (loaded === "all" && (requested === "wasm" || requested === "webgl")) {
    return true;
  }
  return false;
}
async function importOrtBundle(bundle) {
  switch (bundle) {
    case "jspi":
      return import('onnxruntime-web/jspi');
    case "webgpu":
      return import('onnxruntime-web/webgpu');
    case "webgl":
      return import('onnxruntime-web/webgl');
    case "all":
      return import('onnxruntime-web/all');
    case "wasm":
    default:
      return import('onnxruntime-web/wasm');
  }
}
function applyOnnxRuntimeWebOptions(ort2, options) {
  if (options.wasmPaths !== void 0) {
    ort2.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.numThreads !== void 0) {
    ort2.env.wasm.numThreads = options.numThreads;
  }
  if (options.proxy !== void 0) {
    ort2.env.wasm.proxy = options.proxy;
  }
}
function resolveRequestedBundle(options) {
  var _a, _b, _c, _d;
  const executionProviders = (_c = (_b = (_a = options.sessionOptions) == null ? void 0 : _a.executionProviders) != null ? _b : options.executionProviders) != null ? _c : ["wasm"];
  return resolveOrtBundle(executionProviders, (_d = options.ortBundle) != null ? _d : "auto");
}
async function initializeOnnxRuntimeWeb(options = {}) {
  const bundle = resolveRequestedBundle(options);
  if (ortModule && loadedBundle) {
    if (canReuseOrtBundle(loadedBundle, bundle)) {
      applyOnnxRuntimeWebOptions(ortModule, options);
      return ortModule;
    }
    const error = new Error(
      `onnxruntime-web is already loaded as "${loadedBundle}", but "${bundle}" was requested. Reload the page before switching between WebGPU (native) and WebNN/WebGL (all/JSEP) bundles.`
    );
    error.code = "ORT_BUNDLE_RELOAD_REQUIRED";
    error.loadedBundle = loadedBundle;
    error.requestedBundle = bundle;
    throw error;
  }
  if (!loadingPromise) {
    loadingPromise = importOrtBundle(bundle).then((module2) => {
      ortModule = module2;
      loadedBundle = bundle;
      loadingPromise = null;
      return module2;
    });
  }
  const module = await loadingPromise;
  applyOnnxRuntimeWebOptions(module, options);
  return module;
}
async function ensureOnnxRuntimeWebInitialized(options = {}) {
  return initializeOnnxRuntimeWeb(options);
}
function getOrt() {
  if (!ortModule) {
    throw new Error("ONNX Runtime Web is not initialized. Call Yolo.create() / load() first.");
  }
  return ortModule;
}
function getLoadedOrtBundle() {
  return loadedBundle;
}
var WEBGPU_SESSION_BUSY = /another WebGPU EP inference session is being created/i;
var webGpuSessionLock = Promise.resolve();
function sessionUsesWebGpu(options) {
  var _a;
  const providers = (_a = options == null ? void 0 : options.executionProviders) != null ? _a : [];
  return providers.some((provider) => {
    if (typeof provider === "string") {
      return provider === "webgpu";
    }
    return provider.name === "webgpu";
  });
}
function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
function createInferenceSession(model, options) {
  if (typeof model === "string") {
    return ort.InferenceSession.create(model, options);
  }
  if (model instanceof Uint8Array) {
    return ort.InferenceSession.create(model, options);
  }
  return ort.InferenceSession.create(model, options);
}
async function createOrtInferenceSession(model, options) {
  if (!sessionUsesWebGpu(options)) {
    return createInferenceSession(model, options);
  }
  let release;
  const previous = webGpuSessionLock;
  webGpuSessionLock = new Promise((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    let lastError;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        return await createInferenceSession(model, options);
      } catch (error) {
        lastError = error;
        const message = error instanceof Error ? error.message : String(error);
        if (!WEBGPU_SESSION_BUSY.test(message) || attempt === 5) {
          throw error;
        }
        await delay(40 * (attempt + 1));
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  } finally {
    release();
  }
}
var ort = new Proxy({}, {
  get(_target, property, receiver) {
    return Reflect.get(getOrt(), property, receiver);
  }
});

// src/types.ts
var YoloExecutionProviderNames = [
  "coreml",
  "cpu",
  "cuda",
  "dml",
  "nnapi",
  "tensorrt",
  "wasm",
  "webgl",
  "webgpu",
  "webnn",
  "qnn",
  "xnnpack"
];
var YoloExecutionProviderOptions = [
  { value: "coreml", label: "CoreML" },
  { value: "cpu", label: "CPU" },
  { value: "cuda", label: "CUDA" },
  { value: "dml", label: "DirectML" },
  { value: "nnapi", label: "NNAPI" },
  { value: "tensorrt", label: "TensorRT" },
  { value: "wasm", label: "WASM" },
  { value: "webgl", label: "WebGL" },
  { value: "webgpu", label: "WebGPU" },
  { value: "webnn", label: "WebNN" },
  { value: "qnn", label: "QNN" },
  { value: "xnnpack", label: "XNNPACK" }
];
var YoloWebExecutionProviderOptions = [
  { value: "webgpu", label: "WebGPU" },
  { value: "wasm", label: "WASM" },
  { value: "webnn", label: "WebNN" },
  { value: "webgl", label: "WebGL" },
  { value: "cpu", label: "CPU" }
];
var TrackingInfo = class {
  constructor(options = {}) {
    this.id = options.id;
    this.tail = options.tail;
  }
};
var ObjectDetection = class extends TrackingInfo {
  constructor(options) {
    super({ id: options.id, tail: options.tail });
    this.label = options.label;
    this.confidence = options.confidence;
    this.boundingBox = options.boundingBox;
  }
};
var OBBDetection = class extends ObjectDetection {
  constructor(options) {
    super(options);
    this.orientationAngle = options.orientationAngle;
  }
};
var Segmentation = class extends ObjectDetection {
  constructor(options) {
    var _a;
    super(options);
    this.bitPackedPixelMask = options.bitPackedPixelMask;
    this.segmentationEdgePoints = (_a = options.segmentationEdgePoints) != null ? _a : [];
    this.pixelMaskWidth = options.pixelMaskWidth;
    this.pixelMaskHeight = options.pixelMaskHeight;
  }
};
var PoseEstimation = class extends ObjectDetection {
  constructor(options) {
    super(options);
    this.keyPoints = options.keyPoints;
  }
};
var Classification = class {
  constructor(label, confidence) {
    this.label = label;
    this.confidence = confidence;
  }
};

// src/draw-tool.ts
var DEFAULT_BOX_COLORS = [
  "#22c55e",
  "#3b82f6",
  "#f97316",
  "#e11d48",
  "#8b5cf6",
  "#14b8a6",
  "#f59e0b",
  "#06b6d4"
];
var DEFAULT_POSE_CONNECTIONS = [
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
  [2, 4]
];
var EDGE_NEIGHBOR_OFFSETS = [
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
  { x: -1, y: 1 },
  { x: -1, y: 0 },
  { x: -1, y: -1 },
  { x: 0, y: -1 },
  { x: 1, y: -1 }
];
var DEFAULT_EDGE_FILL_OPACITY = 64;
var pooledMaskCanvas = null;
var pooledMaskContext = null;
var pooledMaskImageData = null;
function getPooledMaskTarget(width, height) {
  if (!pooledMaskCanvas || !pooledMaskContext) {
    pooledMaskCanvas = document.createElement("canvas");
    pooledMaskContext = pooledMaskCanvas.getContext("2d");
    if (!pooledMaskContext) {
      throw new Error("Canvas 2D context is not available.");
    }
  }
  if (pooledMaskCanvas.width < width || pooledMaskCanvas.height < height) {
    pooledMaskCanvas.width = Math.max(width, pooledMaskCanvas.width);
    pooledMaskCanvas.height = Math.max(height, pooledMaskCanvas.height);
    pooledMaskImageData = null;
  }
  if (!pooledMaskImageData || pooledMaskImageData.width !== width || pooledMaskImageData.height !== height) {
    pooledMaskImageData = pooledMaskContext.createImageData(width, height);
  } else {
    pooledMaskImageData.data.fill(0);
  }
  return { canvas: pooledMaskCanvas, context: pooledMaskContext, imageData: pooledMaskImageData };
}
var DrawTool = class {
  static drawObjectDetections(source, detections, canvas, options = {}) {
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    this.drawBoundingBoxes(context, detections, width, height, options);
  }
  static drawClassifications(source, classifications, canvas, options = {}) {
    var _a, _b, _c, _d, _e;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = (_a = options.font) != null ? _a : `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const fontColor = (_b = options.fontColor) != null ? _b : "#f8fafc";
    const backgroundColor = (_c = options.backgroundColor) != null ? _c : "rgba(15, 23, 42, 0.72)";
    const drawConfidenceScore = (_d = options.drawConfidenceScore) != null ? _d : true;
    const drawLabelBackground = (_e = options.drawLabelBackground) != null ? _e : true;
    const margin = 10;
    const lineGap = 8;
    context.font = font;
    context.textBaseline = "top";
    const lineHeight = this.getCanvasFontSize(font) + lineGap;
    const labels = classifications.map((item) => `${item.label}${drawConfidenceScore ? ` (${(item.confidence * 100).toFixed(1)}%)` : ""}`);
    const boxWidth = Math.max(0, ...labels.map((label) => context.measureText(label).width)) + margin * 2;
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
  static drawObbDetections(source, detections, canvas, options = {}) {
    var _a, _b, _c, _d, _e, _f, _g;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = (_a = options.font) != null ? _a : `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const lineWidth = (_b = options.lineWidth) != null ? _b : Math.max(2, Math.round(Math.min(width, height) / 320));
    const drawLabel = (_c = options.drawLabel) != null ? _c : true;
    const drawConfidenceScore = (_d = options.drawConfidenceScore) != null ? _d : true;
    const drawLabelBackground = (_e = options.drawLabelBackground) != null ? _e : true;
    const colors = (_f = options.boundingBoxHexColors) != null ? _f : [...DEFAULT_BOX_COLORS];
    const alpha = this.getDetectionDrawingAlpha(options);
    const fontColor = this.withAlpha((_g = options.fontColor) != null ? _g : "#f8fafc", alpha);
    context.font = font;
    context.textBaseline = "middle";
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
          fontColor
        });
      }
    }
  }
  static drawSegmentations(source, segmentations, canvas, options = {}) {
    var _a, _b, _c, _d, _e, _f;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const colors = (_a = options.boundingBoxHexColors) != null ? _a : [...DEFAULT_BOX_COLORS];
    const drawMask = (_b = options.drawSegmentationPixelMask) != null ? _b : true;
    const drawContour = (_c = options.drawContour) != null ? _c : false;
    const drawBoundingBoxes = (_d = options.drawBoundingBoxes) != null ? _d : true;
    const pixelMaskOpacity = (_e = options.pixelMaskOpacity) != null ? _e : 128;
    if (drawMask) {
      for (const segmentation of segmentations) {
        this.drawSegmentationMask(context, segmentation, this.getDetectionColor(segmentation, colors, void 0, pixelMaskOpacity));
      }
    }
    if (drawContour) {
      for (let index = 0; index < segmentations.length; index += 1) {
        const segmentation = segmentations[index];
        this.drawSegmentationContour(
          context,
          segmentation,
          this.getDetectionColor(segmentation, colors, options.strokeStyle),
          (_f = options.contourThickness) != null ? _f : 2
        );
      }
    }
    if (drawBoundingBoxes || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, segmentations, width, height, options);
    }
  }
  static drawPoseEstimations(source, poseEstimations, canvas, options = {}) {
    var _a, _b, _c, _d, _e, _f, _g;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const confidence = (_a = options.poseConfidence) != null ? _a : 0.25;
    const defaultPoseColor = (_b = options.defaultPoseColor) != null ? _b : "#22c55e";
    const radius = (_c = options.keyPointRadius) != null ? _c : Math.max(3, Math.round(Math.min(width, height) / 260));
    const lineWidth = (_d = options.lineWidth) != null ? _d : Math.max(2, Math.round(Math.min(width, height) / 360));
    const markers = options.keyPointMarkers;
    context.lineWidth = lineWidth;
    for (const pose of poseEstimations) {
      this.drawPoseConnections(context, pose.keyPoints, confidence, markers, defaultPoseColor);
      for (let i = 0; i < pose.keyPoints.length; i += 1) {
        const keyPoint = pose.keyPoints[i];
        if (keyPoint.confidence < confidence) {
          continue;
        }
        context.fillStyle = (_f = (_e = markers == null ? void 0 : markers[i]) == null ? void 0 : _e.color) != null ? _f : defaultPoseColor;
        context.beginPath();
        context.arc(keyPoint.x, keyPoint.y, radius, 0, Math.PI * 2);
        context.fill();
      }
    }
    if (((_g = options.drawBoundingBoxes) != null ? _g : true) || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, poseEstimations, width, height, options);
    }
  }
  static extractSegmentationEdgePoints(segmentation) {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const destWidth = right - left;
    const destHeight = bottom - top;
    const { width: maskWidth, height: maskHeight } = this.resolvePackedMaskSize(segmentation);
    if (destWidth <= 0 || destHeight <= 0 || segmentation.bitPackedPixelMask.byteLength === 0 || maskWidth <= 0) {
      return [];
    }
    const edgeKeys = /* @__PURE__ */ new Set();
    for (let y = 0; y < maskHeight; y += 1) {
      for (let x = 0; x < maskWidth; x += 1) {
        const pixelIndex = y * maskWidth + x;
        if (!this.isSegmentationEdgePixel(segmentation.bitPackedPixelMask, pixelIndex, x, y, maskWidth, maskHeight)) {
          continue;
        }
        edgeKeys.add(pixelIndex);
      }
    }
    const local = this.traceOrderedEdgePoints(edgeKeys, 0, 0, maskWidth);
    const scaleX = destWidth / maskWidth;
    const scaleY = destHeight / maskHeight;
    return local.map((point) => ({
      x: left + point.x * scaleX,
      y: top + point.y * scaleY
    }));
  }
  static extractSegmentationsEdgePoints(segmentations) {
    return segmentations.map((segmentation) => this.extractSegmentationEdgePoints(segmentation));
  }
  static extractSegmentationContours(segmentation) {
    return this.splitEdgeContours(this.extractSegmentationEdgePoints(segmentation));
  }
  static splitEdgeContours(points) {
    if (points.length === 0) {
      return [];
    }
    const maxGap = this.estimateEdgeStep(points) * 1.51;
    const contours = [];
    let current = [];
    let previous = null;
    for (const point of points) {
      const gap = previous ? Math.max(Math.abs(point.x - previous.x), Math.abs(point.y - previous.y)) : Number.POSITIVE_INFINITY;
      if (!previous || gap > maxGap) {
        if (current.length >= 3) {
          contours.push(current);
        }
        current = [point];
      } else {
        current.push(point);
      }
      previous = point;
    }
    if (current.length >= 3) {
      contours.push(current);
    }
    return contours;
  }
  static estimateEdgeStep(points) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
    let minStep = Number.POSITIVE_INFINITY;
    const limit = Math.min(points.length, 256);
    for (let index = 1; index < limit; index += 1) {
      const step = Math.max(
        Math.abs(((_b = (_a = points[index]) == null ? void 0 : _a.x) != null ? _b : 0) - ((_d = (_c = points[index - 1]) == null ? void 0 : _c.x) != null ? _d : 0)),
        Math.abs(((_f = (_e = points[index]) == null ? void 0 : _e.y) != null ? _f : 0) - ((_h = (_g = points[index - 1]) == null ? void 0 : _g.y) != null ? _h : 0))
      );
      if (step > 1e-6 && step < minStep) {
        minStep = step;
      }
    }
    return Number.isFinite(minStep) ? minStep : 1;
  }
  static drawPackedMaskOverlay(context, segmentation, options = {}) {
    var _a, _b, _c, _d;
    const box = (_a = options.dest) != null ? _a : segmentation.boundingBox;
    const destWidth = box.right - box.left;
    const destHeight = box.bottom - box.top;
    const { width: maskWidth, height: maskHeight } = this.resolvePackedMaskSize(segmentation);
    if (destWidth <= 0 || destHeight <= 0 || segmentation.bitPackedPixelMask.byteLength === 0 || maskWidth <= 0) {
      return;
    }
    const { canvas: maskCanvas, context: maskContext, imageData } = getPooledMaskTarget(maskWidth, maskHeight);
    const fill = this.parseCanvasColor((_b = options.fill) != null ? _b : "rgba(34, 197, 94, 0.35)");
    const stroke = this.parseCanvasColor((_d = (_c = options.stroke) != null ? _c : options.fill) != null ? _d : "#22c55e");
    const pixels = imageData.data;
    const total = maskWidth * maskHeight;
    const packed = segmentation.bitPackedPixelMask;
    for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
      if (!this.isPackedMaskSet(packed, pixelIndex)) {
        continue;
      }
      const x = pixelIndex % maskWidth;
      const y = pixelIndex / maskWidth | 0;
      const edge = this.isSegmentationEdgePixel(packed, pixelIndex, x, y, maskWidth, maskHeight);
      const color = edge ? stroke : fill;
      const offset = pixelIndex * 4;
      pixels[offset] = color.r;
      pixels[offset + 1] = color.g;
      pixels[offset + 2] = color.b;
      pixels[offset + 3] = edge ? Math.max(color.a, 200) : fill.a;
    }
    maskContext.putImageData(imageData, 0, 0);
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(maskCanvas, 0, 0, maskWidth, maskHeight, box.left, box.top, destWidth, destHeight);
    context.restore();
  }
  static traceOrderedEdgePoints(edgeKeys, left, top, width) {
    const remaining = new Set(edgeKeys);
    const ordered = [];
    while (remaining.size > 0) {
      const startKey = this.getTopLeftKey(remaining, width);
      const contour = this.traceEdgeComponent(startKey, remaining, width);
      for (const key of contour) {
        ordered.push({
          x: left + key % width,
          y: top + Math.floor(key / width)
        });
      }
    }
    return ordered;
  }
  static traceEdgeComponent(startKey, remaining, width) {
    const contour = [];
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
  static getNextEdgeNeighbor(currentKey, previousKey, directionIndex, remaining, width) {
    const currentX = currentKey % width;
    const currentY = Math.floor(currentKey / width);
    const preferredDirection = previousKey >= 0 ? this.getDirectionIndex(previousKey, currentKey, width) : directionIndex;
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
  static getDirectionIndex(fromKey, toKey, width) {
    const fromX = fromKey % width;
    const fromY = Math.floor(fromKey / width);
    const toX = toKey % width;
    const toY = Math.floor(toKey / width);
    const dx = Math.sign(toX - fromX);
    const dy = Math.sign(toY - fromY);
    const index = EDGE_NEIGHBOR_OFFSETS.findIndex((offset) => offset.x === dx && offset.y === dy);
    return index >= 0 ? index : 0;
  }
  static getTopLeftKey(keys, width) {
    let topLeftKey = -1;
    let topLeftX = Number.POSITIVE_INFINITY;
    let topLeftY = Number.POSITIVE_INFINITY;
    for (const key of keys) {
      const x = key % width;
      const y = Math.floor(key / width);
      if (y < topLeftY || y === topLeftY && x < topLeftX) {
        topLeftKey = key;
        topLeftX = x;
        topLeftY = y;
      }
    }
    return topLeftKey;
  }
  static drawBoundingBoxes(context, detections, width, height, options = {}) {
    var _a, _b, _c, _d, _e, _f, _g;
    const colors = (_a = options.boundingBoxHexColors) != null ? _a : [...DEFAULT_BOX_COLORS];
    const lineWidth = (_b = options.lineWidth) != null ? _b : Math.max(2, Math.round(Math.min(width, height) / 320));
    const font = (_c = options.font) != null ? _c : `${Math.max(14, Math.round(Math.min(width, height) / 70))}px Arial`;
    const drawLabel = (_d = options.drawLabel) != null ? _d : true;
    const drawConfidenceScore = (_e = options.drawConfidenceScore) != null ? _e : true;
    const drawLabelBackground = (_f = options.drawLabelBackground) != null ? _f : true;
    const alpha = this.getDetectionDrawingAlpha(options);
    const fontColor = this.withAlpha((_g = options.fontColor) != null ? _g : "#f8fafc", alpha);
    context.lineWidth = lineWidth;
    context.font = font;
    context.textBaseline = "middle";
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
          fontColor
        });
      }
    }
  }
  static prepareDrawingCanvas(source, canvas, drawSource = true) {
    const { width, height } = this.getImageSourceSize(source);
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Canvas 2D context is not available.");
    }
    if (canvas.width !== width) {
      canvas.width = width;
    }
    if (canvas.height !== height) {
      canvas.height = height;
    }
    context.clearRect(0, 0, width, height);
    if (drawSource) {
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(source, 0, 0, width, height);
    }
    return { context, width, height };
  }
  static getDetectionColor(detection, colors, fallback, alpha = 255) {
    var _a;
    const color = (_a = fallback != null ? fallback : colors[detection.label.index % colors.length]) != null ? _a : DEFAULT_BOX_COLORS[0];
    return this.withAlpha(color, alpha);
  }
  static getDetectionDrawingAlpha(options) {
    var _a;
    if (options.resultOpacity !== void 0) {
      return Math.round(this.clamp(options.resultOpacity, 0, 1) * 255);
    }
    return (_a = options.boundingBoxOpacity) != null ? _a : 255;
  }
  static withAlpha(color, alpha) {
    if (!color.startsWith("#") || color.length !== 7) {
      return color;
    }
    const r = Number.parseInt(color.slice(1, 3), 16);
    const g = Number.parseInt(color.slice(3, 5), 16);
    const b = Number.parseInt(color.slice(5, 7), 16);
    const normalizedAlpha = this.clamp(alpha, 0, 255) / 255;
    return `rgba(${r}, ${g}, ${b}, ${normalizedAlpha})`;
  }
  static drawDetectionLabel(context, detection, x, y, backgroundColor, options) {
    var _a;
    const fontSize = this.getCanvasFontSize(options.font);
    const margin = Math.max(4, Math.round(fontSize / 3));
    const label = `${detection.label.name}${options.drawConfidenceScore ? ` ${(detection.confidence * 100).toFixed(1)}%` : ""}`;
    const textWidth = context.measureText(label).width;
    const boxWidth = textWidth + margin * 2;
    const boxHeight = fontSize + margin * 2;
    const left = this.clamp(Math.round(x), 0, Math.max(0, context.canvas.width - boxWidth));
    const top = this.clamp(Math.round(y), 0, Math.max(0, context.canvas.height - boxHeight));
    context.font = options.font;
    context.textBaseline = "middle";
    if (options.drawLabelBackground) {
      context.fillStyle = backgroundColor;
      context.fillRect(left, top, boxWidth, boxHeight);
    }
    context.fillStyle = (_a = options.fontColor) != null ? _a : "#f8fafc";
    context.fillText(label, left + margin, top + boxHeight / 2);
  }
  static getCanvasFontSize(font) {
    const match = font.match(/(\d+(?:\.\d+)?)px/);
    return match ? Number(match[1]) : 14;
  }
  static getObbCorners(box, radians) {
    const centerX = (box.left + box.right) / 2;
    const centerY = (box.top + box.bottom) / 2;
    const cos = Math.cos(radians);
    const sin = Math.sin(radians);
    const corners = [
      { x: box.left, y: box.top },
      { x: box.right, y: box.top },
      { x: box.right, y: box.bottom },
      { x: box.left, y: box.bottom }
    ];
    return corners.map((point) => {
      const dx = point.x - centerX;
      const dy = point.y - centerY;
      return {
        x: centerX + dx * cos - dy * sin,
        y: centerY + dx * sin + dy * cos
      };
    });
  }
  static drawSegmentationMask(context, segmentation, color) {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const destWidth = right - left;
    const destHeight = bottom - top;
    const { width: maskWidth, height: maskHeight } = this.resolvePackedMaskSize(segmentation);
    if (destWidth <= 0 || destHeight <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }
    const { canvas: maskCanvas, context: maskContext, imageData } = getPooledMaskTarget(maskWidth, maskHeight);
    const rgba = this.parseCanvasColor(color);
    const total = maskWidth * maskHeight;
    const pixels = imageData.data;
    for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
      if (!this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex)) {
        continue;
      }
      const offset = pixelIndex * 4;
      pixels[offset] = rgba.r;
      pixels[offset + 1] = rgba.g;
      pixels[offset + 2] = rgba.b;
      pixels[offset + 3] = rgba.a;
    }
    maskContext.putImageData(imageData, 0, 0);
    context.drawImage(maskCanvas, 0, 0, maskWidth, maskHeight, left, top, destWidth, destHeight);
  }
  static drawSegmentationContour(context, segmentation, color, thickness) {
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
  static drawSegmentationEdgePoints(source, segmentations, canvas, options = {}) {
    var _a, _b, _c, _d;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const colors = (_a = options.boundingBoxHexColors) != null ? _a : [...DEFAULT_BOX_COLORS];
    const thickness = (_b = options.contourThickness) != null ? _b : 2;
    const drawBoundingBoxes = (_c = options.drawBoundingBoxes) != null ? _c : true;
    const alpha = this.getDetectionDrawingAlpha(options);
    const fillBaseOpacity = (_d = options.pixelMaskOpacity) != null ? _d : DEFAULT_EDGE_FILL_OPACITY;
    const fillOpacity = options.resultOpacity !== void 0 ? Math.round(fillBaseOpacity * this.clamp(options.resultOpacity, 0, 1)) : fillBaseOpacity;
    for (let index = 0; index < segmentations.length; index += 1) {
      const segmentation = segmentations[index];
      const contours = this.extractSegmentationContours(segmentation);
      const strokeColor = this.getDetectionColor(segmentation, colors, options.strokeStyle, alpha);
      const fillColor = this.getDetectionColor(
        segmentation,
        colors,
        options.fillStyle,
        fillOpacity
      );
      if (options.drawSegmentationPixelMask === true) {
        if (options.fillSegmentationEdgePoints === true) {
          this.drawSegmentationMask(context, segmentation, fillColor);
        }
        this.drawOrderedEdgeContours(context, contours, strokeColor, thickness);
      }
    }
    if (drawBoundingBoxes || options.drawLabel !== false) {
      this.drawBoundingBoxes(context, segmentations, width, height, options);
    }
  }
  static drawOrderedEdgeContours(context, contours, strokeColor, thickness, fillColor) {
    if (contours.length === 0) {
      return;
    }
    context.save();
    context.lineWidth = thickness;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.strokeStyle = strokeColor;
    for (const contour of contours) {
      if (contour.length === 0) {
        continue;
      }
      context.beginPath();
      context.moveTo(contour[0].x, contour[0].y);
      for (let index = 1; index < contour.length; index += 1) {
        context.lineTo(contour[index].x, contour[index].y);
      }
      if (fillColor && this.isMostlyClosedContour(contour)) {
        context.closePath();
        context.fillStyle = fillColor;
        context.fill();
      }
      context.stroke();
    }
    context.restore();
  }
  static isMostlyClosedContour(points) {
    if (points.length < 3) {
      return false;
    }
    const first = points[0];
    const last = points[points.length - 1];
    const gap = Math.max(Math.abs(first.x - last.x), Math.abs(first.y - last.y));
    return gap <= Math.max(this.estimateEdgeStep(points) * 3, 4);
  }
  static resolvePackedMaskSize(segmentation) {
    const destWidth = Math.max(1, segmentation.boundingBox.right - segmentation.boundingBox.left);
    const destHeight = Math.max(1, segmentation.boundingBox.bottom - segmentation.boundingBox.top);
    if (segmentation.pixelMaskWidth && segmentation.pixelMaskHeight) {
      return {
        width: Math.max(1, Math.round(segmentation.pixelMaskWidth)),
        height: Math.max(1, Math.round(segmentation.pixelMaskHeight))
      };
    }
    const expectedBytes = Math.ceil(destWidth * destHeight / 8);
    if (segmentation.bitPackedPixelMask.byteLength >= expectedBytes) {
      return { width: destWidth, height: destHeight };
    }
    const packedBits = segmentation.bitPackedPixelMask.byteLength * 8;
    const aspect = destWidth / destHeight;
    const maskHeight = Math.max(1, Math.round(Math.sqrt(packedBits / Math.max(aspect, 1e-6))));
    const maskWidth = Math.max(1, Math.round(maskHeight * aspect));
    return { width: maskWidth, height: maskHeight };
  }
  static isSegmentationEdgePixel(mask, pixelIndex, x, y, width, height) {
    if (!this.isPackedMaskSet(mask, pixelIndex)) {
      return false;
    }
    return x === 0 || x === width - 1 || y === 0 || y === height - 1 || !this.isPackedMaskSet(mask, pixelIndex - 1) || !this.isPackedMaskSet(mask, pixelIndex + 1) || !this.isPackedMaskSet(mask, pixelIndex - width) || !this.isPackedMaskSet(mask, pixelIndex + width);
  }
  static isPackedMaskSet(mask, pixelIndex) {
    if (pixelIndex < 0) {
      return false;
    }
    const byteIndex = pixelIndex >> 3;
    if (byteIndex >= mask.byteLength) {
      return false;
    }
    return (mask[byteIndex] & 1 << (pixelIndex & 7)) !== 0;
  }
  static parseCanvasColor(color) {
    var _a, _b, _c, _d;
    if (color.startsWith("#") && color.length === 7) {
      return {
        r: Number.parseInt(color.slice(1, 3), 16),
        g: Number.parseInt(color.slice(3, 5), 16),
        b: Number.parseInt(color.slice(5, 7), 16),
        a: 255
      };
    }
    const rgba = color.match(/rgba?\(([^)]+)\)/);
    if (rgba) {
      const parts = rgba[1].split(",").map((part) => Number(part.trim()));
      return {
        r: (_a = parts[0]) != null ? _a : 34,
        g: (_b = parts[1]) != null ? _b : 197,
        b: (_c = parts[2]) != null ? _c : 94,
        a: Math.round(((_d = parts[3]) != null ? _d : 1) * 255)
      };
    }
    return { r: 34, g: 197, b: 94, a: 128 };
  }
  static drawPoseConnections(context, keyPoints, confidence, markers, defaultColor) {
    var _a, _b, _c;
    if (markers && markers.length > 0) {
      for (let sourceIndex = 0; sourceIndex < markers.length; sourceIndex += 1) {
        const source = keyPoints[sourceIndex];
        if (!source || source.confidence < confidence) {
          continue;
        }
        for (const connection of (_b = (_a = markers[sourceIndex]) == null ? void 0 : _a.connections) != null ? _b : []) {
          this.drawPoseConnection(context, source, keyPoints[connection.index], confidence, (_c = connection.color) != null ? _c : defaultColor);
        }
      }
      return;
    }
    for (const [sourceIndex, targetIndex] of DEFAULT_POSE_CONNECTIONS) {
      this.drawPoseConnection(context, keyPoints[sourceIndex], keyPoints[targetIndex], confidence, defaultColor);
    }
  }
  static drawPoseConnection(context, source, target, confidence, color) {
    if (!source || !target || source.confidence < confidence || target.confidence < confidence) {
      return;
    }
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(source.x, source.y);
    context.lineTo(target.x, target.y);
    context.stroke();
  }
  static getImageSourceSize(img) {
    if (img instanceof HTMLImageElement) {
      return {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height
      };
    }
    if (img instanceof HTMLVideoElement) {
      return {
        width: img.videoWidth || img.width,
        height: img.videoHeight || img.height
      };
    }
    if ("displayWidth" in img && "displayHeight" in img) {
      return {
        width: img.displayWidth || img.codedWidth,
        height: img.displayHeight || img.codedHeight
      };
    }
    if (img instanceof SVGImageElement) {
      const width = img.width.baseVal.value || img.getBoundingClientRect().width;
      const height = img.height.baseVal.value || img.getBoundingClientRect().height;
      return { width, height };
    }
    return {
      width: img.width,
      height: img.height
    };
  }
  static clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }
};

// src/onnx-model.ts
var MODEL_METADATA_PROPS_FIELD = 14;
var STRING_ENTRY_KEY_FIELD = 1;
var STRING_ENTRY_VALUE_FIELD = 2;
async function parseOnnxModel(session, model, options = {}) {
  var _a, _b, _c;
  const customMetaData = await parseCustomMetadata(model);
  const inputShapes = getShapes(session.inputMetadata);
  const outputShapes = getShapes(session.outputMetadata);
  const firstInputShape = (_a = Object.values(inputShapes)[0]) != null ? _a : [];
  const labels = options.labels ? mapLabels(parseLabelsInput(options.labels)) : customMetaData.names ? mapLabelsAndColors(customMetaData.names) : inferLabels(outputShapes);
  return {
    inputShapes,
    outputShapes,
    customMetaData,
    modelDataType: getModelDataType(session.inputMetadata),
    modelType: (_b = options.modelType) != null ? _b : getModelType(customMetaData, outputShapes),
    modelVersion: (_c = options.modelVersion) != null ? _c : getModelVersion(customMetaData, outputShapes),
    labels,
    inputShapeSize: calculateTotalInputShapeSize(firstInputShape)
  };
}
async function parseCustomMetadata(model) {
  const bytes = await getModelBytes(model);
  const metadata = {};
  for (const field of readFields(bytes)) {
    if (field.fieldNumber !== MODEL_METADATA_PROPS_FIELD || field.wireType !== 2 || !field.value) {
      continue;
    }
    const entry = parseStringStringEntry(field.value);
    if (entry.key) {
      metadata[entry.key] = entry.value;
    }
  }
  return metadata;
}
async function getModelBytes(model) {
  if (typeof model === "string") {
    const response = await fetch(model);
    if (!response.ok) {
      throw new Error(`Failed to fetch ONNX model metadata: ${response.status} ${response.statusText}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  }
  if (model instanceof Uint8Array) {
    return model;
  }
  return new Uint8Array(model);
}
function getShapes(metadata) {
  const shapes = {};
  for (const item of metadata) {
    shapes[item.name] = item.isTensor ? item.shape.map(normalizeDimension) : [];
  }
  return shapes;
}
function normalizeDimension(dimension) {
  return typeof dimension === "number" ? dimension : -1;
}
function getModelDataType(metadata) {
  const firstTensor = metadata.find((item) => item.isTensor);
  return (firstTensor == null ? void 0 : firstTensor.isTensor) && firstTensor.type === "float16" ? "Float16" : "Float";
}
function getModelType(metadata, outputShapes) {
  if (metadata.task) {
    return getModelTypeFromMetadata(metadata.task);
  }
  if (isRfdetrSegmentationOutput(outputShapes)) {
    return "Segmentation";
  }
  if (isRfdetrOutput(outputShapes)) {
    return "ObjectDetection";
  }
  throw new Error("Unsupported task");
}
function getModelTypeFromMetadata(modelType) {
  switch (modelType) {
    case "classify":
      return "Classification";
    case "detect":
      return "ObjectDetection";
    case "obb":
      return "ObbDetection";
    case "segment":
      return "Segmentation";
    case "pose":
      return "PoseEstimation";
    default:
      throw new Error("Unsupported task");
  }
}
function getModelVersion(metadata, outputShapes) {
  if (metadata.description) {
    return getModelVersionFromDescription(metadata.description);
  }
  if (isRfdetrOutput(outputShapes)) {
    return "RFDETR";
  }
  throw new Error("Onnx model not supported!");
}
function getModelVersionFromDescription(modelDescription) {
  const version = modelDescription.toLowerCase();
  if (version.startsWith("ultralytics yolov5")) return "V5U";
  if (version.startsWith("ultralytics yolov8")) return "V8";
  if (version.startsWith("ultralytics yoloe-v8")) return "V8E";
  if (version.startsWith("ultralytics yolov9")) return "V9";
  if (version.startsWith("ultralytics yolov10")) return "V10";
  if (version.startsWith("ultralytics yolo11")) return "V11";
  if (version.startsWith("ultralytics yoloe-11")) return "V11E";
  if (version.startsWith("ultralytics yolov12")) return "V12";
  if (version.startsWith("ultralytics yolo26")) return "V26";
  if (version.includes("worldv2")) return "WORLDV2";
  if (version.startsWith("ultralytics rt-detr")) return "RTDETR";
  if (version.startsWith("ultralytics") && !version.includes("yolo")) return "V8";
  throw new Error("Onnx model not supported!");
}
function mapLabelsAndColors(onnxLabelData) {
  const labels = parseLabels(onnxLabelData);
  return Object.entries(labels).map(([, name], index) => ({
    index,
    name
  }));
}
function mapLabels(labels) {
  return labels.map((name, index) => ({ index, name }));
}
function parseLabelsInput(labels) {
  if (typeof labels !== "string") {
    return labels.map((label) => label.trim()).filter(Boolean);
  }
  const content = labels.trim();
  if (!content) {
    return [];
  }
  try {
    const parsed = JSON.parse(content);
    if (Array.isArray(parsed)) {
      return parsed.map(String).map((label) => label.trim()).filter(Boolean);
    }
  } catch (e) {
  }
  return content.split(/\r?\n|,/).map((label) => label.trim()).filter(Boolean);
}
function inferLabels(outputShapes) {
  const classCount = tryGetRfdetrClassCount(outputShapes);
  if (classCount === null) {
    throw new Error('ONNX custom metadata "names" is missing. Pass labels in YoloOptions for metadata-free models.');
  }
  return Array.from({ length: classCount }, (_, index) => ({
    index,
    name: index === 0 ? "background_class" : `class_${index}`
  }));
}
function parseLabels(onnxLabelData) {
  const labels = {};
  const content = onnxLabelData.trim().replace(/^\{|\}$/g, "");
  if (!content) {
    return labels;
  }
  for (const item of content.split(/\s*,\s*/)) {
    const match = item.match(/^\s*'?(\d+)'?\s*:\s*['"]?(.+?)['"]?\s*$/);
    if (!match) {
      continue;
    }
    labels[Number(match[1])] = match[2];
  }
  return labels;
}
function calculateTotalInputShapeSize(shape) {
  if (shape.length === 0) {
    return 0;
  }
  let shapeSize = 1;
  for (const dimension of shape) {
    if (dimension <= 0) {
      throw new Error(`All shape dimensions must be positive. Found invalid value: ${dimension}`);
    }
    shapeSize *= dimension;
  }
  return shapeSize;
}
function isRfdetrOutput(outputShapes) {
  const dets = outputShapes.dets;
  const labels = outputShapes.labels;
  return Boolean(dets && labels && dets.length === 3 && labels.length === 3 && dets[2] === 4);
}
function isRfdetrSegmentationOutput(outputShapes) {
  const masks = outputShapes.masks;
  return Boolean(isRfdetrOutput(outputShapes) && masks && masks.length === 4);
}
function tryGetRfdetrClassCount(outputShapes) {
  return isRfdetrOutput(outputShapes) ? outputShapes.labels[2] : null;
}
function parseStringStringEntry(bytes) {
  let key = "";
  let value = "";
  for (const field of readFields(bytes)) {
    if (field.wireType !== 2 || !field.value) {
      continue;
    }
    if (field.fieldNumber === STRING_ENTRY_KEY_FIELD) {
      key = decodeUtf8(field.value);
    } else if (field.fieldNumber === STRING_ENTRY_VALUE_FIELD) {
      value = decodeUtf8(field.value);
    }
  }
  return { key, value };
}
function readFields(bytes) {
  const state = { offset: 0 };
  const fields = [];
  while (state.offset < bytes.length) {
    const tag = readVarint(bytes, state);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 7;
    fields.push({
      fieldNumber,
      wireType,
      value: readFieldValue(bytes, state, wireType)
    });
  }
  return fields;
}
function readFieldValue(bytes, state, wireType) {
  switch (wireType) {
    case 0:
      readVarint(bytes, state);
      return void 0;
    case 1:
      state.offset += 8;
      return void 0;
    case 2: {
      const length = readVarint(bytes, state);
      const start = state.offset;
      state.offset += length;
      return bytes.subarray(start, start + length);
    }
    case 5:
      state.offset += 4;
      return void 0;
    default:
      throw new Error(`Unsupported ONNX protobuf wire type: ${wireType}`);
  }
}
function readVarint(bytes, state) {
  let result = 0;
  let shift = 0;
  while (state.offset < bytes.length) {
    const byte = bytes[state.offset++];
    result += (byte & 127) * 2 ** shift;
    if ((byte & 128) === 0) {
      return result;
    }
    shift += 7;
  }
  throw new Error("Invalid ONNX protobuf varint.");
}
function decodeUtf8(bytes) {
  return new TextDecoder().decode(bytes);
}

// src/handler/common.ts
function unsupportedTask(task) {
  throw new Error(`${task} is not supported by this YOLO model.`);
}
function toDetection(object) {
  return {
    label: object.label,
    confidence: object.confidence,
    boundingBox: object.boundingBox
  };
}
function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}
function scalePoint(x, y, input) {
  var _a, _b, _c, _d;
  const roiLeft = (_b = (_a = input.roi) == null ? void 0 : _a.left) != null ? _b : 0;
  const roiTop = (_d = (_c = input.roi) == null ? void 0 : _c.top) != null ? _d : 0;
  return {
    x: roiLeft + clamp(Math.trunc((x - input.xPad) * input.gain), 0, input.sourceWidth - 1),
    y: roiTop + clamp(Math.trunc((y - input.yPad) * input.gain), 0, input.sourceHeight - 1)
  };
}
function removeOverlappingBoxes(detections, iouThreshold) {
  if (detections.length === 0 || iouThreshold <= 0) {
    return detections;
  }
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const results = [];
  for (const detection of sorted) {
    const overlaps = results.some((item) => calculateIoU(detection.boundingBox, item.boundingBox) > iouThreshold);
    if (!overlaps) {
      results.push(detection);
    }
  }
  return results;
}
function calculateIoU(a, b) {
  const left = Math.max(a.left, b.left);
  const top = Math.max(a.top, b.top);
  const right = Math.min(a.right, b.right);
  const bottom = Math.min(a.bottom, b.bottom);
  const width = right - left;
  const height = bottom - top;
  if (width <= 0 || height <= 0) {
    return 0;
  }
  const intersection = width * height;
  const areaA = (a.right - a.left) * (a.bottom - a.top);
  const areaB = (b.right - b.left) * (b.bottom - b.top);
  return intersection / (areaA + areaB - intersection);
}
function packSegmentationMask(proto, maskWeights, maskWidth, maskHeight, crop, targetWidth, targetHeight, pixelConfidence) {
  var _a, _b;
  if (targetWidth <= 0 || targetHeight <= 0 || maskWidth <= 0 || maskHeight <= 0) {
    return new Uint8Array();
  }
  const totalPixels = targetWidth * targetHeight;
  const packed = new Uint8Array(Math.ceil(totalPixels / 8));
  const planeSize = maskWidth * maskHeight;
  const cropLeft = clamp(Math.trunc(crop.left), 0, maskWidth - 1);
  const cropTop = clamp(Math.trunc(crop.top), 0, maskHeight - 1);
  const cropRight = clamp(Math.trunc(crop.right), cropLeft + 1, maskWidth);
  const cropBottom = clamp(Math.trunc(crop.bottom), cropTop + 1, maskHeight);
  const cropWidth = cropRight - cropLeft;
  const cropHeight = cropBottom - cropTop;
  const activeWeights = [];
  const activeOffsets = [];
  for (let channel = 0; channel < maskWeights.length; channel += 1) {
    const weight = (_a = maskWeights[channel]) != null ? _a : 0;
    if (weight === 0) {
      continue;
    }
    activeWeights.push(weight);
    activeOffsets.push(channel * planeSize);
  }
  if (activeWeights.length === 0) {
    return packed;
  }
  const lowResMask = new Float32Array(cropWidth * cropHeight);
  for (let y = 0; y < cropHeight; y += 1) {
    const protoRowOffset = (cropTop + y) * maskWidth + cropLeft;
    const maskRowOffset = y * cropWidth;
    for (let x = 0; x < cropWidth; x += 1) {
      const protoOffset = protoRowOffset + x;
      let pixelWeight = 0;
      for (let channel = 0; channel < activeWeights.length; channel += 1) {
        pixelWeight += ((_b = proto[activeOffsets[channel] + protoOffset]) != null ? _b : 0) * activeWeights[channel];
      }
      lowResMask[maskRowOffset + x] = sigmoid(pixelWeight);
    }
  }
  if (targetWidth === cropWidth && targetHeight === cropHeight) {
    for (let pixelIndex = 0; pixelIndex < totalPixels; pixelIndex += 1) {
      if (lowResMask[pixelIndex] > pixelConfidence) {
        packed[pixelIndex >> 3] |= 1 << (pixelIndex & 7);
      }
    }
    return packed;
  }
  const x0Lookup = new Int32Array(targetWidth);
  const x1Lookup = new Int32Array(targetWidth);
  const xWeightLookup = new Float32Array(targetWidth);
  const xScale = cropWidth / targetWidth;
  const yScale = cropHeight / targetHeight;
  for (let targetX = 0; targetX < targetWidth; targetX += 1) {
    const sourceX = (targetX + 0.5) * xScale - 0.5;
    const x0 = clamp(Math.floor(sourceX), 0, cropWidth - 1);
    const x1 = x0 < cropWidth - 1 ? x0 + 1 : x0;
    x0Lookup[targetX] = x0;
    x1Lookup[targetX] = x1;
    xWeightLookup[targetX] = sourceX - x0;
  }
  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = (targetY + 0.5) * yScale - 0.5;
    const y0 = clamp(Math.floor(sourceY), 0, cropHeight - 1);
    const y1 = y0 < cropHeight - 1 ? y0 + 1 : y0;
    const yWeight = sourceY - y0;
    const row0 = y0 * cropWidth;
    const row1 = y1 * cropWidth;
    const targetRow = targetY * targetWidth;
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const x0 = x0Lookup[targetX];
      const x1 = x1Lookup[targetX];
      const xWeight = xWeightLookup[targetX];
      const topLeft = lowResMask[row0 + x0];
      const topRight = lowResMask[row0 + x1];
      const bottomLeft = lowResMask[row1 + x0];
      const bottomRight = lowResMask[row1 + x1];
      const top = topLeft + (topRight - topLeft) * xWeight;
      const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
      if (top + (bottom - top) * yWeight > pixelConfidence) {
        const pixelIndex = targetRow + targetX;
        packed[pixelIndex >> 3] |= 1 << (pixelIndex & 7);
      }
    }
  }
  return packed;
}
function downscaleBoxToMask(box, maskWidth, maskHeight, inputWidth, inputHeight) {
  const scalingFactorW = maskWidth / inputWidth;
  const scalingFactorH = maskHeight / inputHeight;
  return {
    left: clamp(Math.floor(box.left * scalingFactorW), 0, maskWidth - 1),
    top: clamp(Math.floor(box.top * scalingFactorH), 0, maskHeight - 1),
    right: clamp(Math.ceil(box.right * scalingFactorW), 0, maskWidth - 1),
    bottom: clamp(Math.ceil(box.bottom * scalingFactorH), 0, maskHeight - 1)
  };
}

// src/handler/yolo26/yolo26-hanlder.ts
var Yolo26Handler = class {
  constructor(yolo) {
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    return this._yolo.preprocessImage(img, roi);
  }
  async RunObjectDetection(img, confidence, iou, roi = null) {
    this.ensureTask("ObjectDetection");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const detections = this.decodeFlatDetections(output.data, input, confidence);
    return detections.map((item) => new ObjectDetection(toDetection(item)));
  }
  async RunObbDetection(img, confidence, iou, roi = null) {
    this.ensureTask("ObbDetection");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const data = Object.values(result)[0].data;
    const stride = this.getStride();
    const detections = [];
    for (let i = 0; i + 6 < data.length; i += stride) {
      const score = data[i + 4];
      if (score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];
      if (!label) {
        continue;
      }
      const x = data[i];
      const y = data[i + 1];
      const halfWidth = data[i + 2] / 2;
      const halfHeight = data[i + 3] / 2;
      detections.push(new OBBDetection({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(
          x - halfWidth,
          y - halfHeight,
          x + halfWidth,
          y + halfHeight,
          input
        ),
        orientationAngle: data[i + 6]
      }));
    }
    return detections;
  }
  async RunSegmentation(img, confidence, pixelConfidence, iou, roi = null) {
    var _a, _b;
    this.ensureTask("Segmentation");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const outputs = Object.values(result);
    const data = outputs[0].data;
    const maskData = (_a = outputs[1]) == null ? void 0 : _a.data;
    const outputShapes = Object.values(this._yolo.onnxModel.outputShapes);
    const maskShape = outputShapes[1];
    if (!maskData || !maskShape || maskShape.length < 4) {
      throw new Error(`Unsupported YOLO26 segmentation output shapes: ${JSON.stringify(outputShapes)}`);
    }
    const stride = this.getStride();
    const maskChannels = maskShape[1];
    const maskHeight = maskShape[2];
    const maskWidth = maskShape[3];
    const inputWidth = input.inputShape[3];
    const inputHeight = input.inputShape[2];
    const segmentations = [];
    for (let i = 0; i + 5 < data.length; i += stride) {
      const score = data[i + 4];
      if (score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];
      if (!label) {
        continue;
      }
      const boundingBoxUnscaled = { left: data[i], top: data[i + 1], right: data[i + 2], bottom: data[i + 3] };
      const boundingBox = this._yolo.scaleBoundingBox(
        boundingBoxUnscaled.left,
        boundingBoxUnscaled.top,
        boundingBoxUnscaled.right,
        boundingBoxUnscaled.bottom,
        input
      );
      const maskWeights = [];
      for (let channel = 0; channel < maskChannels; channel += 1) {
        maskWeights.push((_b = data[i + 6 + channel]) != null ? _b : 0);
      }
      const bitPackedPixelMask = packSegmentationMask(
        maskData,
        maskWeights,
        maskWidth,
        maskHeight,
        downscaleBoxToMask(boundingBoxUnscaled, maskWidth, maskHeight, inputWidth, inputHeight),
        boundingBox.right - boundingBox.left,
        boundingBox.bottom - boundingBox.top,
        pixelConfidence
      );
      segmentations.push(new Segmentation({
        label,
        confidence: score,
        boundingBox,
        bitPackedPixelMask
      }));
    }
    return segmentations;
  }
  async RunPoseEstimation(img, confidence, iou, roi = null) {
    var _a;
    this.ensureTask("PoseEstimation");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const data = Object.values(result)[0].data;
    const stride = this.getStride();
    const dimensions = 6;
    const keypointDimensions = 3;
    const totalKeypoints = Math.floor((stride - dimensions) / keypointDimensions);
    const detections = [];
    for (let i = 0; i + dimensions < data.length; i += stride) {
      const score = data[i + 4];
      if (score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];
      if (!label) {
        continue;
      }
      const keyPoints = [];
      for (let keypoint = 0; keypoint < totalKeypoints; keypoint += 1) {
        const offset = i + dimensions + keypoint * keypointDimensions;
        const point = scalePoint(data[offset], data[offset + 1], input);
        keyPoints.push({
          x: point.x,
          y: point.y,
          confidence: (_a = data[offset + 2]) != null ? _a : 0
        });
      }
      detections.push(new PoseEstimation({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], input),
        keyPoints
      }));
    }
    return detections;
  }
  async RunClassification(img, classes) {
    this.ensureTask("Classification");
    const input = this.preprocessImage(img, null);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const data = Array.from(output.data);
    return data.map((confidence, index) => {
      var _a, _b;
      return {
        label: (_b = (_a = this._yolo.onnxModel.labels[index]) == null ? void 0 : _a.name) != null ? _b : String(index),
        confidence
      };
    }).sort((a, b) => b.confidence - a.confidence).slice(0, classes).map((item) => new Classification(item.label, item.confidence));
  }
  decodeFlatDetections(data, input, confidence) {
    const detections = [];
    for (let i = 0; i + 5 < data.length; i += 6) {
      const score = data[i + 4];
      if (score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];
      if (!label) {
        continue;
      }
      detections.push({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], input),
        boundingBoxUnscaled: { left: data[i], top: data[i + 1], right: data[i + 2], bottom: data[i + 3] },
        boundingBoxIndex: i
      });
    }
    return detections;
  }
  getStride() {
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];
    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported YOLO26 output shape: ${JSON.stringify(outputShape)}`);
    }
    return outputShape[2];
  }
  ensureTask(task) {
    if (this._yolo.onnxModel.modelType !== task) {
      unsupportedTask(task);
    }
  }
};

// src/handler/yolov10/yolov10-handler.ts
var Yolov10Handler = class {
  constructor(yolo) {
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    return this._yolo.preprocessImage(img, roi);
  }
  async RunObjectDetection(img, confidence, iou, roi = null) {
    if (this._yolo.onnxModel.modelType !== "ObjectDetection") {
      unsupportedTask("ObjectDetection");
    }
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const data = Object.values(result)[0].data;
    const detections = [];
    for (let i = 0; i + 5 < data.length; i += 6) {
      const score = data[i + 4];
      if (score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];
      if (!label) {
        continue;
      }
      detections.push(new ObjectDetection({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], input)
      }));
    }
    return detections;
  }
  RunObbDetection(img, confidence, iou, roi = null) {
    unsupportedTask("ObbDetection");
  }
  RunSegmentation(img, confidence, pixelConfidence, iou, roi = null) {
    unsupportedTask("Segmentation");
  }
  RunPoseEstimation(img, confidence, iou, roi = null) {
    unsupportedTask("PoseEstimation");
  }
  RunClassification(img, classes) {
    unsupportedTask("Classification");
  }
};

// src/handler/yolov8/yolov8-handler.ts
var Yolov8Handler = class {
  constructor(yolo) {
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    return this._yolo.preprocessImage(img, roi);
  }
  async RunObjectDetection(img, confidence, iou, roi = null) {
    this.ensureTask("ObjectDetection");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const objects = this.decodeObjectDetections(output.data, input, confidence, iou);
    return objects.map((item) => new ObjectDetection(toDetection(item)));
  }
  async RunObbDetection(img, confidence, iou, roi = null) {
    this.ensureTask("ObbDetection");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const objects = this.decodeObjectDetections(Object.values(result)[0].data, input, confidence, iou);
    return objects.map((item) => {
      var _a;
      return new OBBDetection({
        ...toDetection(item),
        orientationAngle: (_a = item.orientationAngle) != null ? _a : 0
      });
    });
  }
  async RunSegmentation(img, confidence, pixelConfidence, iou, roi = null) {
    var _a, _b;
    this.ensureTask("Segmentation");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const outputs = Object.values(result);
    const detectionData = outputs[0].data;
    const maskData = (_a = outputs[1]) == null ? void 0 : _a.data;
    const objects = this.decodeObjectDetections(detectionData, input, confidence, iou);
    const outputShapes = Object.values(this._yolo.onnxModel.outputShapes);
    const detectionShape = outputShapes[0];
    const maskShape = outputShapes[1];
    if (!maskData || !detectionShape || !maskShape || maskShape.length < 4) {
      throw new Error(`Unsupported YOLOv8 segmentation output shapes: ${JSON.stringify(outputShapes)}`);
    }
    const labels = this._yolo.onnxModel.labels.length;
    const predictions = detectionShape[2];
    const maskChannels = maskShape[1];
    const maskHeight = maskShape[2];
    const maskWidth = maskShape[3];
    const inputWidth = input.inputShape[3];
    const inputHeight = input.inputShape[2];
    const segmentations = [];
    const maskWeights = new Float32Array(maskChannels);
    for (const item of objects) {
      let maskOffset = item.boundingBoxIndex + predictions * (labels + 4);
      for (let channel = 0; channel < maskChannels; channel += 1, maskOffset += predictions) {
        maskWeights[channel] = (_b = detectionData[maskOffset]) != null ? _b : 0;
      }
      const targetWidth = item.boundingBox.right - item.boundingBox.left;
      const targetHeight = item.boundingBox.bottom - item.boundingBox.top;
      const crop = downscaleBoxToMask(item.boundingBoxUnscaled, maskWidth, maskHeight, inputWidth, inputHeight);
      const bitPackedPixelMask = packSegmentationMask(
        maskData,
        maskWeights,
        maskWidth,
        maskHeight,
        crop,
        targetWidth,
        targetHeight,
        pixelConfidence
      );
      segmentations.push(new Segmentation({
        ...toDetection(item),
        bitPackedPixelMask
      }));
    }
    return segmentations;
  }
  async RunPoseEstimation(img, confidence, iou, roi = null) {
    this.ensureTask("PoseEstimation");
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const data = output.data;
    const objects = this.decodeObjectDetections(data, input, confidence, iou);
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];
    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported YOLOv8 pose output shape: ${JSON.stringify(outputShape)}`);
    }
    const inputChannels = input.inputShape[1];
    const modelOutputElements = outputShape[1];
    const modelOutputChannels = outputShape[2];
    const labels = this._yolo.onnxModel.labels.length;
    const totalKeypoints = Math.floor(modelOutputElements / inputChannels) - labels;
    return objects.map((item) => {
      var _a;
      const keyPoints = [];
      let keypointOffset = item.boundingBoxIndex + modelOutputChannels * (4 + labels);
      for (let keypoint = 0; keypoint < totalKeypoints; keypoint += 1) {
        const xIndex = keypointOffset;
        const yIndex = xIndex + modelOutputChannels;
        const cIndex = yIndex + modelOutputChannels;
        keypointOffset += modelOutputChannels * 3;
        const point = scalePoint(data[xIndex], data[yIndex], input);
        keyPoints.push({
          x: point.x,
          y: point.y,
          confidence: (_a = data[cIndex]) != null ? _a : 0
        });
      }
      return new PoseEstimation({
        ...toDetection(item),
        keyPoints
      });
    });
  }
  async RunClassification(img, classes) {
    this.ensureTask("Classification");
    const input = this.preprocessImage(img, null);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const data = Array.from(output.data);
    return data.map((confidence, index) => {
      var _a, _b;
      return {
        label: (_b = (_a = this._yolo.onnxModel.labels[index]) == null ? void 0 : _a.name) != null ? _b : String(index),
        confidence
      };
    }).sort((a, b) => b.confidence - a.confidence).slice(0, classes).map((item) => new Classification(item.label, item.confidence));
  }
  decodeObjectDetections(data, input, confidence, iou) {
    var _a;
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];
    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported YOLOv8 output shape: ${JSON.stringify(outputShape)}`);
    }
    const predictions = outputShape[2];
    const labels = this._yolo.onnxModel.labels.length;
    const attribute2 = predictions * 2;
    const attribute3 = predictions * 3;
    const attribute4 = predictions * 4;
    const objects = [];
    for (let i = 0; i < predictions; i += 1) {
      let labelOffset = i + attribute4;
      let bestConfidence = 0;
      let bestLabelIndex = -1;
      for (let labelIndex = 0; labelIndex < labels; labelIndex += 1, labelOffset += predictions) {
        const boxConfidence = data[labelOffset];
        if (boxConfidence > bestConfidence) {
          bestConfidence = boxConfidence;
          bestLabelIndex = labelIndex;
        }
      }
      if (bestConfidence < confidence || bestLabelIndex < 0) {
        continue;
      }
      const x = data[i];
      const y = data[i + predictions];
      const w = data[i + attribute2];
      const h = data[i + attribute3];
      const label = this._yolo.onnxModel.labels[bestLabelIndex];
      if (!label) {
        continue;
      }
      const x1 = x - w / 2;
      const y1 = y - h / 2;
      const x2 = x + w / 2;
      const y2 = y + h / 2;
      objects.push({
        label,
        confidence: bestConfidence,
        boundingBox: this._yolo.scaleBoundingBox(x1, y1, x2, y2, input),
        boundingBoxUnscaled: { left: x1, top: y1, right: x2, bottom: y2 },
        boundingBoxIndex: i,
        orientationAngle: (_a = data[i + predictions * (4 + labels)]) != null ? _a : 0
      });
    }
    return removeOverlappingBoxes(objects, iou);
  }
  ensureTask(task) {
    if (this._yolo.onnxModel.modelType !== task) {
      unsupportedTask(task);
    }
  }
};

// src/handler/RT-DETR/RT-DETR-handler.ts
var RT_DETRHandler = class {
  constructor(yolo) {
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    return this._yolo.preprocessImage(img, roi);
  }
  async RunObjectDetection(img, confidence, iou, roi = null) {
    var _a, _b, _c, _d;
    if (this._yolo.onnxModel.modelType !== "ObjectDetection") {
      unsupportedTask("ObjectDetection");
    }
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
    const output = Object.values(result)[0];
    const data = output.data;
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];
    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported RT-DETR output shape: ${JSON.stringify(outputShape)}`);
    }
    const predictions = outputShape[1];
    const attributes = outputShape[2];
    const totalLabels = this._yolo.onnxModel.labels.length;
    const inputWidth = input.inputShape[3];
    const inputHeight = input.inputShape[2];
    const detections = [];
    for (let prediction = 0; prediction < predictions; prediction += 1) {
      const offset = prediction * attributes;
      const x = data[offset];
      const y = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];
      let labelIndex = -1;
      let score = 0;
      for (let c = 0; c < totalLabels; c += 1) {
        const classScore = data[offset + 4 + c];
        if (classScore > score) {
          score = classScore;
          labelIndex = c;
        }
      }
      if (labelIndex < 0 || score < confidence) {
        continue;
      }
      const label = this._yolo.onnxModel.labels[labelIndex];
      if (!label) {
        continue;
      }
      const halfWidth = w * inputWidth * 0.5 * input.gain;
      const halfHeight = h * inputHeight * 0.5 * input.gain;
      const centerX = (x * inputWidth - input.xPad) * input.gain;
      const centerY = (y * inputHeight - input.yPad) * input.gain;
      const roiLeft = (_b = (_a = input.roi) == null ? void 0 : _a.left) != null ? _b : 0;
      const roiTop = (_d = (_c = input.roi) == null ? void 0 : _c.top) != null ? _d : 0;
      detections.push(new ObjectDetection({
        label,
        confidence: score,
        boundingBox: {
          left: roiLeft + this.clamp(Math.trunc(centerX - halfWidth), 0, input.sourceWidth - 1),
          top: roiTop + this.clamp(Math.trunc(centerY - halfHeight), 0, input.sourceHeight - 1),
          right: roiLeft + this.clamp(Math.trunc(centerX + halfWidth), 0, input.sourceWidth - 1),
          bottom: roiTop + this.clamp(Math.trunc(centerY + halfHeight), 0, input.sourceHeight - 1)
        }
      }));
    }
    return detections;
  }
  RunObbDetection(img, confidence, iou, roi = null) {
    unsupportedTask("ObbDetection");
  }
  RunSegmentation(img, confidence, pixelConfidence, iou, roi = null) {
    unsupportedTask("Segmentation");
  }
  RunPoseEstimation(img, confidence, iou, roi = null) {
    unsupportedTask("PoseEstimation");
  }
  RunClassification(img, classes) {
    unsupportedTask("Classification");
  }
  clamp(value, min, max) {
    return clamp(value, min, max);
  }
};

// src/handler/RF-DETR/RF-DETR-handler.ts
var BACKGROUND_CLASS_PREFIX = "background_class";
var DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406];
var DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225];
var RF_DETRHandler = class {
  constructor(yolo) {
    this.preprocessCanvas = null;
    this.preprocessContext = null;
    this.preprocessTensorData = null;
    this.preprocessTensorSize = 0;
    this.x0Lookup = null;
    this.x1Lookup = null;
    this.xWeightLookup = null;
    this.y0Lookup = null;
    this.y1Lookup = null;
    this.yWeightLookup = null;
    this.interpolationCacheKey = "";
    this.topIndices = null;
    this.topScores = null;
    this.webGpuPipeline = null;
    this.webGpuDevice = null;
    this.webGpuFallbackWarned = false;
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    var _a, _b, _c;
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;
    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = (_a = this._yolo.yoloOptions.imageResize) != null ? _a : "stretch";
    const { drawWidth, drawHeight, xPad, yPad, gain } = resizeMode === "stretch" ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 } : this.calculateProportionalResize(sourceRect.width, sourceRect.height, modelWidth, modelHeight);
    const context = this.getPreprocessContext(modelWidth, modelHeight);
    const coversCanvas = xPad <= 0 && yPad <= 0 && drawWidth >= modelWidth && drawHeight >= modelHeight;
    if (!coversCanvas) {
      context.clearRect(0, 0, modelWidth, modelHeight);
    }
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      xPad,
      yPad,
      drawWidth,
      drawHeight
    );
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight).data;
    const pixelCount = modelWidth * modelHeight;
    const tensorData = this.getPreprocessTensorData(channels * pixelCount);
    const imageMean = (_b = this._yolo.yoloOptions.imageMean) != null ? _b : DEFAULT_IMAGE_MEAN;
    const imageStd = (_c = this._yolo.yoloOptions.imageStd) != null ? _c : DEFAULT_IMAGE_STD;
    this.writeCanvasTensor(
      imageData,
      modelWidth,
      modelHeight,
      drawWidth,
      drawHeight,
      xPad,
      yPad,
      imageMean,
      imageStd,
      tensorData
    );
    return {
      tensorData,
      inputName: this._yolo.inputNames[0],
      inputShape,
      sourceWidth: sourceRect.width,
      sourceHeight: sourceRect.height,
      xPad,
      yPad,
      gain,
      resizeMode,
      roi
    };
  }
  async preprocessImageForRun(img, roi = null) {
    if (this._yolo.preprocessBackend !== "webgpu") {
      return this.preprocessImage(img, roi);
    }
    try {
      return await this.preprocessImageWebGpu(img, roi);
    } catch (error) {
      if (!this.webGpuFallbackWarned) {
        console.warn("[RF-DETR] WebGPU preprocessing failed. Falling back to CPU preprocessing.", error);
        this.webGpuFallbackWarned = true;
      }
      return this.preprocessImage(img, roi);
    }
  }
  async preprocessImageWebGpu(img, roi = null) {
    var _a, _b, _c;
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;
    if (channels !== 3) {
      return this.preprocessImage(img, roi);
    }
    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = (_a = this._yolo.yoloOptions.imageResize) != null ? _a : "stretch";
    const { drawWidth, drawHeight, xPad, yPad, gain } = resizeMode === "stretch" ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 } : this.calculateProportionalResize(sourceRect.width, sourceRect.height, modelWidth, modelHeight);
    const context = this.getPreprocessContext(modelWidth, modelHeight);
    const coversCanvas = xPad <= 0 && yPad <= 0 && drawWidth >= modelWidth && drawHeight >= modelHeight;
    if (!coversCanvas) {
      context.clearRect(0, 0, modelWidth, modelHeight);
    }
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      xPad,
      yPad,
      drawWidth,
      drawHeight
    );
    const imageMean = (_b = this._yolo.yoloOptions.imageMean) != null ? _b : DEFAULT_IMAGE_MEAN;
    const imageStd = (_c = this._yolo.yoloOptions.imageStd) != null ? _c : DEFAULT_IMAGE_STD;
    const inputTensor = await this.createWebGpuInputTensor(context.canvas, inputShape, imageMean, imageStd);
    return {
      tensorData: new Float32Array(0),
      inputTensor,
      inputName: this._yolo.inputNames[0],
      inputShape,
      sourceWidth: sourceRect.width,
      sourceHeight: sourceRect.height,
      xPad,
      yPad,
      gain,
      resizeMode,
      roi
    };
  }
  async createWebGpuInputTensor(canvas, inputShape, imageMean, imageStd) {
    const [, channels, height, width] = inputShape;
    const device = await this._yolo.getWebGpuDevice();
    const usage = globalThis.GPUBufferUsage;
    const textureUsage = globalThis.GPUTextureUsage;
    const outputByteLength = channels * width * height * Float32Array.BYTES_PER_ELEMENT;
    const texture = device.createTexture({
      size: [width, height, 1],
      format: "rgba8unorm",
      usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT
    });
    const outputBuffer = device.createBuffer({
      size: outputByteLength,
      usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST
    });
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: usage.UNIFORM | usage.COPY_DST
    });
    const params = new Float32Array([
      width,
      height,
      0,
      0,
      1 / imageStd[0],
      1 / imageStd[1],
      1 / imageStd[2],
      0,
      -imageMean[0] / imageStd[0],
      -imageMean[1] / imageStd[1],
      -imageMean[2] / imageStd[2],
      0
    ]);
    device.queue.copyExternalImageToTexture(
      { source: canvas },
      { texture },
      { width, height }
    );
    device.queue.writeBuffer(paramsBuffer, 0, params);
    const pipeline = this.getWebGpuPreprocessPipeline(device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);
    texture.destroy();
    paramsBuffer.destroy();
    return this._yolo.tensorFromGpuBuffer(outputBuffer, inputShape, () => outputBuffer.destroy());
  }
  getWebGpuPreprocessPipeline(device) {
    if (this.webGpuPipeline && this.webGpuDevice === device) {
      return this.webGpuPipeline;
    }
    this.webGpuDevice = device;
    this.webGpuPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: `
struct Params {
  size: vec4<f32>,
  scale: vec4<f32>,
  bias: vec4<f32>,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(params.size.x);
  let height = u32(params.size.y);

  if (id.x >= width || id.y >= height) {
    return;
  }

  let rgba = textureLoad(inputTexture, vec2<i32>(i32(id.x), i32(id.y)), 0);
  let pixel = id.y * width + id.x;
  let planeSize = width * height;

  outputTensor[pixel] = rgba.r * params.scale.x + params.bias.x;
  outputTensor[planeSize + pixel] = rgba.g * params.scale.y + params.bias.y;
  outputTensor[planeSize * 2u + pixel] = rgba.b * params.scale.z + params.bias.z;
}
          `
        }),
        entryPoint: "main"
      }
    });
    return this.webGpuPipeline;
  }
  async RunObjectDetection(img, confidence, iou, roi = null) {
    var _a, _b;
    if (this._yolo.onnxModel.modelType !== "ObjectDetection") {
      unsupportedTask("ObjectDetection");
    }
    const input = await this.preprocessImageForRun(img, roi);
    const result = await this.runWithPreprocessedInput(input);
    const dets = (_a = result.dets) == null ? void 0 : _a.data;
    const labels = (_b = result.labels) == null ? void 0 : _b.data;
    if (!dets || !labels) {
      throw new Error(`Unsupported RF-DETR outputs: ${Object.keys(result).join(", ")}`);
    }
    return this.decodeObjectDetections(dets, labels, input, confidence);
  }
  RunObbDetection(img, confidence, iou, roi = null) {
    unsupportedTask("ObbDetection");
  }
  RunSegmentation(img, confidence, pixelConfidence, iou, roi = null) {
    if (this._yolo.onnxModel.modelType !== "Segmentation") {
      unsupportedTask("Segmentation");
    }
    return this.runSegmentation(img, confidence, roi);
  }
  RunPoseEstimation(img, confidence, iou, roi = null) {
    unsupportedTask("PoseEstimation");
  }
  RunClassification(img, classes) {
    unsupportedTask("Classification");
  }
  decodeObjectDetections(dets, logits, input, confidence) {
    const detsShape = this._yolo.onnxModel.outputShapes.dets;
    const labelsShape = this._yolo.onnxModel.outputShapes.labels;
    if (!detsShape || !labelsShape || detsShape.length !== 3 || labelsShape.length !== 3) {
      throw new Error(`Unsupported RF-DETR output shapes: ${JSON.stringify(this._yolo.onnxModel.outputShapes)}`);
    }
    const predictions = detsShape[1];
    const classCount = labelsShape[2];
    const backgroundClassIndex = this._yolo.onnxModel.labels.findIndex(
      (label) => label.name.toLowerCase().startsWith(BACKGROUND_CLASS_PREFIX)
    );
    const labels = this._yolo.onnxModel.labels;
    const { topIndices, topScores, topCount } = this.getRankedCandidates(logits, predictions, classCount);
    const objects = [];
    for (let i = 0; i < topCount; i += 1) {
      const candidateConfidence = topScores[i];
      if (candidateConfidence < confidence) {
        continue;
      }
      const flatIndex = topIndices[i];
      const prediction = Math.trunc(flatIndex / classCount);
      const labelIndex = flatIndex - prediction * classCount;
      if (labelIndex === backgroundClassIndex) {
        continue;
      }
      const label = labels[labelIndex];
      if (!label) {
        continue;
      }
      const boxOffset = prediction * 4;
      const cx = dets[boxOffset];
      const cy = dets[boxOffset + 1];
      const w = dets[boxOffset + 2];
      const h = dets[boxOffset + 3];
      const x1 = cx - w / 2;
      const y1 = cy - h / 2;
      const x2 = cx + w / 2;
      const y2 = cy + h / 2;
      objects.push(new ObjectDetection({
        label,
        confidence: candidateConfidence,
        boundingBox: this.scaleNormalizedBoundingBox(x1, y1, x2, y2, input)
      }));
    }
    return objects;
  }
  async runSegmentation(img, confidence, roi) {
    var _a, _b, _c;
    const input = await this.preprocessImageForRun(img, roi);
    const result = await this.runWithPreprocessedInput(input);
    const dets = (_a = result.dets) == null ? void 0 : _a.data;
    const logits = (_b = result.labels) == null ? void 0 : _b.data;
    const masks = (_c = result.masks) == null ? void 0 : _c.data;
    if (!dets || !logits || !masks) {
      throw new Error(`Unsupported RF-DETR segmentation outputs: ${Object.keys(result).join(", ")}`);
    }
    return this.decodeSegmentations(dets, logits, masks, input, confidence);
  }
  async runWithPreprocessedInput(input) {
    var _a, _b;
    const inputTensor = (_a = input.inputTensor) != null ? _a : this._yolo.tensor("float32", input.tensorData, input.inputShape);
    try {
      return await this._yolo.run({
        [input.inputName]: inputTensor
      });
    } finally {
      (_b = input.inputTensor) == null ? void 0 : _b.dispose();
    }
  }
  decodeSegmentations(dets, logits, masks, input, confidence) {
    const detsShape = this._yolo.onnxModel.outputShapes.dets;
    const labelsShape = this._yolo.onnxModel.outputShapes.labels;
    const masksShape = this._yolo.onnxModel.outputShapes.masks;
    if (!detsShape || !labelsShape || !masksShape || detsShape.length !== 3 || labelsShape.length !== 3 || masksShape.length !== 4) {
      throw new Error(`Unsupported RF-DETR segmentation output shapes: ${JSON.stringify(this._yolo.onnxModel.outputShapes)}`);
    }
    const predictions = detsShape[1];
    const classCount = labelsShape[2];
    const maskHeight = masksShape[2];
    const maskWidth = masksShape[3];
    const maskPlaneSize = maskWidth * maskHeight;
    const backgroundClassIndex = this._yolo.onnxModel.labels.findIndex(
      (label) => label.name.toLowerCase().startsWith(BACKGROUND_CLASS_PREFIX)
    );
    const labels = this._yolo.onnxModel.labels;
    const { topIndices, topScores, topCount } = this.getRankedCandidates(logits, predictions, classCount);
    const segmentations = [];
    for (let i = 0; i < topCount; i += 1) {
      const candidateConfidence = topScores[i];
      if (candidateConfidence < confidence) {
        continue;
      }
      const flatIndex = topIndices[i];
      const prediction = Math.trunc(flatIndex / classCount);
      const labelIndex = flatIndex - prediction * classCount;
      if (labelIndex === backgroundClassIndex) {
        continue;
      }
      const label = labels[labelIndex];
      if (!label) {
        continue;
      }
      const boxOffset = prediction * 4;
      const cx = dets[boxOffset];
      const cy = dets[boxOffset + 1];
      const w = dets[boxOffset + 2];
      const h = dets[boxOffset + 3];
      const x1 = cx - w / 2;
      const y1 = cy - h / 2;
      const x2 = cx + w / 2;
      const y2 = cy + h / 2;
      const boundingBox = this.scaleNormalizedBoundingBox(x1, y1, x2, y2, input);
      const bitPackedPixelMask = this.packRfdetrMask(
        masks,
        prediction * maskPlaneSize,
        maskWidth,
        maskHeight,
        boundingBox,
        input
      );
      segmentations.push(new Segmentation({
        label,
        confidence: candidateConfidence,
        boundingBox,
        bitPackedPixelMask
      }));
    }
    return segmentations;
  }
  getRankedCandidates(logits, predictions, classCount) {
    const maxDetections = predictions;
    const { topIndices, topScores } = this.getTopKBuffers(maxDetections);
    let topCount = 0;
    let minScore = Number.POSITIVE_INFINITY;
    let minPosition = -1;
    for (let prediction = 0; prediction < predictions; prediction += 1) {
      const labelOffset = prediction * classCount;
      for (let labelIndex = 0; labelIndex < classCount; labelIndex += 1) {
        const score = this.sigmoid(logits[labelOffset + labelIndex]);
        const flatIndex = labelOffset + labelIndex;
        if (topCount < maxDetections) {
          topIndices[topCount] = flatIndex;
          topScores[topCount] = score;
          if (score < minScore) {
            minScore = score;
            minPosition = topCount;
          }
          topCount += 1;
          continue;
        }
        if (score <= minScore) {
          continue;
        }
        topIndices[minPosition] = flatIndex;
        topScores[minPosition] = score;
        minScore = topScores[0];
        minPosition = 0;
        for (let i = 1; i < topCount; i += 1) {
          const candidateScore = topScores[i];
          if (candidateScore < minScore) {
            minScore = candidateScore;
            minPosition = i;
          }
        }
      }
    }
    this.sortTopKDescending(topIndices, topScores, topCount);
    return { topIndices, topScores, topCount };
  }
  packRfdetrMask(masks, maskOffset, maskWidth, maskHeight, box, input) {
    const targetWidth = box.right - box.left;
    const targetHeight = box.bottom - box.top;
    if (targetWidth <= 0 || targetHeight <= 0 || maskWidth <= 0 || maskHeight <= 0) {
      return new Uint8Array();
    }
    const totalPixels = targetWidth * targetHeight;
    const packed = new Uint8Array(Math.ceil(totalPixels / 8));
    let cropLeft = 0;
    let cropTop = 0;
    let cropRight = maskWidth;
    let cropBottom = maskHeight;
    if (input.resizeMode !== "stretch") {
      const inputWidth = input.inputShape[3];
      const inputHeight = input.inputShape[2];
      const scale = Math.min(inputWidth / input.sourceWidth, inputHeight / input.sourceHeight);
      const scaledWidth = Math.trunc(input.sourceWidth * scale);
      const scaledHeight = Math.trunc(input.sourceHeight * scale);
      const padX = (inputWidth - scaledWidth) / 2;
      const padY = (inputHeight - scaledHeight) / 2;
      cropLeft = clamp(Math.round(padX * maskWidth / inputWidth), 0, maskWidth - 1);
      cropTop = clamp(Math.round(padY * maskHeight / inputHeight), 0, maskHeight - 1);
      cropRight = clamp(Math.round((padX + scaledWidth) * maskWidth / inputWidth), cropLeft + 1, maskWidth);
      cropBottom = clamp(Math.round((padY + scaledHeight) * maskHeight / inputHeight), cropTop + 1, maskHeight);
    }
    const cropWidth = cropRight - cropLeft;
    const cropHeight = cropBottom - cropTop;
    const xScale = cropWidth / input.sourceWidth;
    const yScale = cropHeight / input.sourceHeight;
    for (let y = 0; y < targetHeight; y += 1) {
      const absoluteY = box.top + y;
      const sourceY = (absoluteY + 0.5) * yScale - 0.5;
      const y0Local = clamp(Math.floor(sourceY), 0, cropHeight - 1);
      const y1Local = y0Local < cropHeight - 1 ? y0Local + 1 : y0Local;
      const yWeight = sourceY - y0Local;
      const row0 = maskOffset + (cropTop + y0Local) * maskWidth;
      const row1 = maskOffset + (cropTop + y1Local) * maskWidth;
      const targetRow = y * targetWidth;
      for (let x = 0; x < targetWidth; x += 1) {
        const absoluteX = box.left + x;
        const sourceX = (absoluteX + 0.5) * xScale - 0.5;
        const x0Local = clamp(Math.floor(sourceX), 0, cropWidth - 1);
        const x1Local = x0Local < cropWidth - 1 ? x0Local + 1 : x0Local;
        const xWeight = sourceX - x0Local;
        const x0 = cropLeft + x0Local;
        const x1 = cropLeft + x1Local;
        const topLeft = masks[row0 + x0];
        const topRight = masks[row0 + x1];
        const bottomLeft = masks[row1 + x0];
        const bottomRight = masks[row1 + x1];
        const top = topLeft + (topRight - topLeft) * xWeight;
        const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
        if (top + (bottom - top) * yWeight > 0) {
          const pixelIndex = targetRow + x;
          packed[pixelIndex >> 3] |= 1 << (pixelIndex & 7);
        }
      }
    }
    return packed;
  }
  getTopKBuffers(size) {
    if (!this.topIndices || this.topIndices.length < size) {
      this.topIndices = new Int32Array(size);
      this.topScores = new Float32Array(size);
    }
    return {
      topIndices: this.topIndices,
      topScores: this.topScores
    };
  }
  sortTopKDescending(indices, scores, length) {
    for (let i = 1; i < length; i += 1) {
      const score = scores[i];
      const index = indices[i];
      let j = i - 1;
      while (j >= 0 && scores[j] < score) {
        scores[j + 1] = scores[j];
        indices[j + 1] = indices[j];
        j -= 1;
      }
      scores[j + 1] = score;
      indices[j + 1] = index;
    }
  }
  sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }
  scaleNormalizedBoundingBox(x1, y1, x2, y2, input) {
    var _a, _b, _c, _d;
    const roiLeft = (_b = (_a = input.roi) == null ? void 0 : _a.left) != null ? _b : 0;
    const roiTop = (_d = (_c = input.roi) == null ? void 0 : _c.top) != null ? _d : 0;
    if (input.resizeMode === "stretch") {
      return {
        left: roiLeft + clamp(Math.trunc(x1 * input.sourceWidth), 0, input.sourceWidth - 1),
        top: roiTop + clamp(Math.trunc(y1 * input.sourceHeight), 0, input.sourceHeight - 1),
        right: roiLeft + clamp(Math.trunc(x2 * input.sourceWidth), 0, input.sourceWidth),
        bottom: roiTop + clamp(Math.trunc(y2 * input.sourceHeight), 0, input.sourceHeight)
      };
    }
    return this._yolo.scaleBoundingBox(
      x1 * input.inputShape[3],
      y1 * input.inputShape[2],
      x2 * input.inputShape[3],
      y2 * input.inputShape[2],
      input
    );
  }
  getInputShape() {
    const shape = Object.values(this._yolo.onnxModel.inputShapes)[0];
    if (!shape || shape.length !== 4) {
      throw new Error(`Unsupported RF-DETR input shape: ${JSON.stringify(shape)}`);
    }
    return [shape[0], shape[1], shape[2], shape[3]];
  }
  writeCanvasTensor(imageData, modelWidth, modelHeight, drawWidth, drawHeight, xPad, yPad, imageMean, imageStd, tensorData) {
    const pixelCount = modelWidth * modelHeight;
    const redScale = 1 / (255 * imageStd[0]);
    const greenScale = 1 / (255 * imageStd[1]);
    const blueScale = 1 / (255 * imageStd[2]);
    const redBias = -imageMean[0] / imageStd[0];
    const greenBias = -imageMean[1] / imageStd[1];
    const blueBias = -imageMean[2] / imageStd[2];
    const left = Math.max(0, Math.trunc(xPad));
    const top = Math.max(0, Math.trunc(yPad));
    const right = Math.min(modelWidth, Math.ceil(xPad + drawWidth));
    const bottom = Math.min(modelHeight, Math.ceil(yPad + drawHeight));
    const canUseUint32 = (imageData.byteOffset & 3) === 0 && (imageData.byteLength & 3) === 0;
    const uint32Pixels = canUseUint32 ? new Uint32Array(imageData.buffer, imageData.byteOffset, pixelCount) : null;
    if (left === 0 && top === 0 && right === modelWidth && bottom === modelHeight) {
      if (uint32Pixels) {
        for (let pixel = 0; pixel < pixelCount; pixel += 1) {
          const rgba = uint32Pixels[pixel];
          tensorData[pixel] = (rgba & 255) * redScale + redBias;
          tensorData[pixelCount + pixel] = (rgba >>> 8 & 255) * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = (rgba >>> 16 & 255) * blueScale + blueBias;
        }
        return;
      }
      for (let imageOffset = 0, pixel = 0; pixel < pixelCount; imageOffset += 4, pixel += 1) {
        tensorData[pixel] = imageData[imageOffset] * redScale + redBias;
        tensorData[pixelCount + pixel] = imageData[imageOffset + 1] * greenScale + greenBias;
        tensorData[pixelCount * 2 + pixel] = imageData[imageOffset + 2] * blueScale + blueBias;
      }
      return;
    }
    tensorData.fill(0);
    for (let y = top; y < bottom; y += 1) {
      let imageOffset = y * modelWidth + left;
      let pixel = y * modelWidth + left;
      if (uint32Pixels) {
        for (let x = left; x < right; x += 1, imageOffset += 1, pixel += 1) {
          const rgba = uint32Pixels[imageOffset];
          tensorData[pixel] = (rgba & 255) * redScale + redBias;
          tensorData[pixelCount + pixel] = (rgba >>> 8 & 255) * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = (rgba >>> 16 & 255) * blueScale + blueBias;
        }
      } else {
        let byteOffset = imageOffset * 4;
        for (let x = left; x < right; x += 1, byteOffset += 4, pixel += 1) {
          tensorData[pixel] = imageData[byteOffset] * redScale + redBias;
          tensorData[pixelCount + pixel] = imageData[byteOffset + 1] * greenScale + greenBias;
          tensorData[pixelCount * 2 + pixel] = imageData[byteOffset + 2] * blueScale + blueBias;
        }
      }
    }
  }
  writePreprocessedTensor(imageData, sourceWidth, sourceHeight, modelWidth, modelHeight, drawWidth, drawHeight, xPad, yPad, imageMean, imageStd, tensorData) {
    const pixelCount = modelWidth * modelHeight;
    tensorData.fill(0);
    const outputLeft = Math.trunc(xPad);
    const outputTop = Math.trunc(yPad);
    const outputWidth = Math.max(1, Math.trunc(drawWidth));
    const outputHeight = Math.max(1, Math.trunc(drawHeight));
    this.ensureInterpolationCache(sourceWidth, sourceHeight, outputWidth, outputHeight);
    const x0Lookup = this.x0Lookup;
    const x1Lookup = this.x1Lookup;
    const xWeightLookup = this.xWeightLookup;
    const y0Lookup = this.y0Lookup;
    const y1Lookup = this.y1Lookup;
    const yWeightLookup = this.yWeightLookup;
    for (let y = 0; y < outputHeight; y += 1) {
      const targetY = outputTop + y;
      if (targetY < 0 || targetY >= modelHeight) {
        continue;
      }
      const y0 = y0Lookup[y];
      const y1 = y1Lookup[y];
      const yWeight = yWeightLookup[y];
      const topRow = y0 * sourceWidth * 4;
      const bottomRow = y1 * sourceWidth * 4;
      for (let x = 0; x < outputWidth; x += 1) {
        const targetX = outputLeft + x;
        if (targetX < 0 || targetX >= modelWidth) {
          continue;
        }
        const x0 = x0Lookup[x];
        const x1 = x1Lookup[x];
        const xWeight = xWeightLookup[x];
        const targetPixel = targetY * modelWidth + targetX;
        const topLeft = topRow + x0 * 4;
        const topRight = topRow + x1 * 4;
        const bottomLeft = bottomRow + x0 * 4;
        const bottomRight = bottomRow + x1 * 4;
        const r = this.interpolate(imageData[topLeft], imageData[topRight], imageData[bottomLeft], imageData[bottomRight], xWeight, yWeight);
        const g = this.interpolate(imageData[topLeft + 1], imageData[topRight + 1], imageData[bottomLeft + 1], imageData[bottomRight + 1], xWeight, yWeight);
        const b = this.interpolate(imageData[topLeft + 2], imageData[topRight + 2], imageData[bottomLeft + 2], imageData[bottomRight + 2], xWeight, yWeight);
        tensorData[targetPixel] = (r / 255 - imageMean[0]) / imageStd[0];
        tensorData[pixelCount + targetPixel] = (g / 255 - imageMean[1]) / imageStd[1];
        tensorData[pixelCount * 2 + targetPixel] = (b / 255 - imageMean[2]) / imageStd[2];
      }
    }
  }
  ensureInterpolationCache(sourceWidth, sourceHeight, outputWidth, outputHeight) {
    const cacheKey = `${sourceWidth}:${sourceHeight}:${outputWidth}:${outputHeight}`;
    if (this.interpolationCacheKey === cacheKey) {
      return;
    }
    this.interpolationCacheKey = cacheKey;
    this.x0Lookup = new Int32Array(outputWidth);
    this.x1Lookup = new Int32Array(outputWidth);
    this.xWeightLookup = new Float32Array(outputWidth);
    this.y0Lookup = new Int32Array(outputHeight);
    this.y1Lookup = new Int32Array(outputHeight);
    this.yWeightLookup = new Float32Array(outputHeight);
    this.fillInterpolationAxis(this.x0Lookup, this.x1Lookup, this.xWeightLookup, outputWidth, sourceWidth);
    this.fillInterpolationAxis(this.y0Lookup, this.y1Lookup, this.yWeightLookup, outputHeight, sourceHeight);
  }
  fillInterpolationAxis(lowerLookup, upperLookup, weightLookup, targetSize, sourceSize) {
    for (let target = 0; target < targetSize; target += 1) {
      const source = (target + 0.5) * (sourceSize / targetSize) - 0.5;
      const index = Math.floor(source);
      if (index < 0) {
        lowerLookup[target] = 0;
        upperLookup[target] = 0;
        weightLookup[target] = 0;
      } else if (index >= sourceSize - 1) {
        lowerLookup[target] = sourceSize - 1;
        upperLookup[target] = sourceSize - 1;
        weightLookup[target] = 0;
      } else {
        lowerLookup[target] = index;
        upperLookup[target] = index + 1;
        weightLookup[target] = source - index;
      }
    }
  }
  interpolate(topLeft, topRight, bottomLeft, bottomRight, xWeight, yWeight) {
    const top = topLeft + (topRight - topLeft) * xWeight;
    const bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
    return top + (bottom - top) * yWeight;
  }
  getSourceRect(img, roi) {
    const { width, height } = this.getImageSourceSize(img);
    if (!roi) {
      return { x: 0, y: 0, width, height };
    }
    const left = clamp(Math.trunc(roi.left), 0, width - 1);
    const top = clamp(Math.trunc(roi.top), 0, height - 1);
    const right = clamp(Math.trunc(roi.right), left + 1, width);
    const bottom = clamp(Math.trunc(roi.bottom), top + 1, height);
    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top
    };
  }
  calculateProportionalResize(sourceWidth, sourceHeight, modelWidth, modelHeight) {
    if (sourceWidth < modelWidth && sourceHeight < modelHeight) {
      return {
        drawWidth: sourceWidth,
        drawHeight: sourceHeight,
        xPad: (modelWidth - sourceWidth) * 0.5,
        yPad: (modelHeight - sourceHeight) * 0.5,
        gain: 1
      };
    }
    const ratio = Math.min(modelWidth / sourceWidth, modelHeight / sourceHeight);
    const drawWidth = sourceWidth * ratio;
    const drawHeight = sourceHeight * ratio;
    return {
      drawWidth,
      drawHeight,
      xPad: (modelWidth - drawWidth) * 0.5,
      yPad: (modelHeight - drawHeight) * 0.5,
      gain: Math.max(sourceWidth / modelWidth, sourceHeight / modelHeight)
    };
  }
  getPreprocessContext(width, height) {
    if (!this.preprocessCanvas) {
      this.preprocessCanvas = document.createElement("canvas");
    }
    if (this.preprocessCanvas.width !== width) {
      this.preprocessCanvas.width = width;
    }
    if (this.preprocessCanvas.height !== height) {
      this.preprocessCanvas.height = height;
    }
    if (!this.preprocessContext) {
      this.preprocessContext = this.preprocessCanvas.getContext("2d", { willReadFrequently: true });
    }
    if (!this.preprocessContext) {
      throw new Error("Canvas 2D context is not available.");
    }
    return this.preprocessContext;
  }
  getPreprocessTensorData(size) {
    if (!this.preprocessTensorData || this.preprocessTensorSize !== size) {
      this.preprocessTensorData = new Float32Array(size);
      this.preprocessTensorSize = size;
    }
    return this.preprocessTensorData;
  }
  getImageSourceSize(img) {
    if (img instanceof HTMLImageElement) {
      return {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height
      };
    }
    if (img instanceof HTMLVideoElement) {
      return {
        width: img.videoWidth || img.width,
        height: img.videoHeight || img.height
      };
    }
    if ("displayWidth" in img && "displayHeight" in img) {
      return {
        width: img.displayWidth || img.codedWidth,
        height: img.displayHeight || img.codedHeight
      };
    }
    if (img instanceof SVGImageElement) {
      const width = img.width.baseVal.value || img.getBoundingClientRect().width;
      const height = img.height.baseVal.value || img.getBoundingClientRect().height;
      return { width, height };
    }
    return {
      width: img.width,
      height: img.height
    };
  }
};

// src/yolo.ts
var DEFAULT_EXECUTION_PROVIDERS = ["wasm"];
var Yolo = class _Yolo {
  constructor(options = {}) {
    this.session = null;
    this._onnxModel = null;
    this._handler = null;
    this.preprocessCanvas = null;
    this.preprocessContext = null;
    this.preprocessTensorData = null;
    this.preprocessTensorSize = 0;
    this.options = options;
    this.model = options.model;
  }
  get yoloOptions() {
    return this.options;
  }
  get preprocessBackend() {
    var _a;
    const requestedBackend = (_a = this.options.preprocessBackend) != null ? _a : this.hasWebGpuExecutionProvider() ? "webgpu" : "cpu";
    if (requestedBackend !== "webgpu") {
      return "cpu";
    }
    return this.hasWebGpuExecutionProvider() ? "webgpu" : "cpu";
  }
  static async create(options) {
    await ensureOnnxRuntimeWebInitialized(options);
    const yolo = new _Yolo(options);
    if (options.model) {
      await yolo.load(options.model);
    }
    return yolo;
  }
  get isLoaded() {
    return this.session !== null;
  }
  get inputNames() {
    return this.ensureSession().inputNames;
  }
  get outputNames() {
    return this.ensureSession().outputNames;
  }
  get onnxModel() {
    if (!this._onnxModel) {
      throw new Error("ONNX model info is not parsed. Call load() first or pass model to Yolo.create().");
    }
    return this._onnxModel;
  }
  async load(model = this.requireModel()) {
    await ensureOnnxRuntimeWebInitialized(this.options);
    await this.dispose();
    this.session = await this.createSession(model);
    this._onnxModel = await parseOnnxModel(this.session, model, this.options);
    var modelVersion = this._onnxModel.modelVersion;
    var modelType = this._onnxModel.modelType;
    if (!this.isSupportedModel(modelVersion, modelType)) {
      throw new Error(`Unsupported model type ${modelType} for model version ${modelVersion}.`);
    }
    switch (modelVersion) {
      case "V5U":
      case "V8":
      case "V8E":
      case "V9":
      case "V11":
      case "V11E":
      case "V12":
      case "WORLDV2":
        this._handler = new Yolov8Handler(this);
        break;
      case "V10":
        this._handler = new Yolov10Handler(this);
        break;
      case "V26":
        this._handler = new Yolo26Handler(this);
        break;
      case "RTDETR":
        this._handler = new RT_DETRHandler(this);
        break;
      case "RFDETR":
        this._handler = new RF_DETRHandler(this);
        break;
      default:
        throw new Error("Unsupported model version: " + modelVersion);
    }
    return this;
  }
  run(feeds, options) {
    return this.ensureSession().run(feeds, options);
  }
  runWithFetches(feeds, fetches, options) {
    return this.ensureSession().run(feeds, fetches, options);
  }
  predict(feeds, options) {
    return this.run(feeds, options);
  }
  RunObjectDetection(img, confidence = 0.2, iou = 0.7, roi = null) {
    return this.ensureHandler().RunObjectDetection(img, confidence, iou, roi);
  }
  RunObbDetection(img, confidence = 0.2, iou = 0.7, roi = null) {
    return this.ensureHandler().RunObbDetection(img, confidence, iou, roi);
  }
  RunSegmentation(img, confidence = 0.2, pixelConfidence = 0.65, iou = 0.7, roi = null) {
    return this.ensureHandler().RunSegmentation(img, confidence, pixelConfidence, iou, roi);
  }
  RunPoseEstimation(img, confidence = 0.2, iou = 0.7, roi = null) {
    return this.ensureHandler().RunPoseEstimation(img, confidence, iou, roi);
  }
  RunClassification(img, classes = 5) {
    return this.ensureHandler().RunClassification(img, classes);
  }
  preprocessImage(img, roi = null) {
    var _a;
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;
    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = (_a = this.options.imageResize) != null ? _a : "proportional";
    const { drawWidth, drawHeight, xPad, yPad, gain } = resizeMode === "stretch" ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 } : this.calculateProportionalResize(
      sourceRect.width,
      sourceRect.height,
      modelWidth,
      modelHeight
    );
    const context = this.getPreprocessContext(modelWidth, modelHeight);
    context.clearRect(0, 0, modelWidth, modelHeight);
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      xPad,
      yPad,
      drawWidth,
      drawHeight
    );
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight).data;
    const pixelCount = modelWidth * modelHeight;
    const tensorData = this.getPreprocessTensorData(channels * pixelCount);
    const imageMean = this.options.imageMean;
    const imageStd = this.options.imageStd;
    for (let i = 0, pixel = 0; i < imageData.length; i += 4, pixel += 1) {
      const r = imageData[i] / 255;
      const g = imageData[i + 1] / 255;
      const b = imageData[i + 2] / 255;
      if (imageMean && imageStd) {
        tensorData[pixel] = (r - imageMean[0]) / imageStd[0];
        tensorData[pixelCount + pixel] = (g - imageMean[1]) / imageStd[1];
        tensorData[pixelCount * 2 + pixel] = (b - imageMean[2]) / imageStd[2];
      } else {
        tensorData[pixel] = r;
        tensorData[pixelCount + pixel] = g;
        tensorData[pixelCount * 2 + pixel] = b;
      }
    }
    return {
      tensorData,
      inputName: this.inputNames[0],
      inputShape,
      sourceWidth: sourceRect.width,
      sourceHeight: sourceRect.height,
      xPad,
      yPad,
      gain,
      resizeMode,
      roi
    };
  }
  scaleBoundingBox(x1, y1, x2, y2, input) {
    var _a, _b, _c, _d;
    const roiLeft = (_b = (_a = input.roi) == null ? void 0 : _a.left) != null ? _b : 0;
    const roiTop = (_d = (_c = input.roi) == null ? void 0 : _c.top) != null ? _d : 0;
    return {
      left: roiLeft + this.clamp(Math.trunc((x1 - input.xPad) * input.gain), 0, input.sourceWidth - 1),
      top: roiTop + this.clamp(Math.trunc((y1 - input.yPad) * input.gain), 0, input.sourceHeight - 1),
      right: roiLeft + this.clamp(Math.trunc((x2 - input.xPad) * input.gain), 0, input.sourceWidth),
      bottom: roiTop + this.clamp(Math.trunc((y2 - input.yPad) * input.gain), 0, input.sourceHeight)
    };
  }
  drawObjectDetections(source, detections, canvas, options = {}) {
    DrawTool.drawObjectDetections(source, detections, canvas, options);
  }
  drawClassifications(source, classifications, canvas, options = {}) {
    DrawTool.drawClassifications(source, classifications, canvas, options);
  }
  drawObbDetections(source, detections, canvas, options = {}) {
    DrawTool.drawObbDetections(source, detections, canvas, options);
  }
  drawSegmentations(source, segmentations, canvas, options = {}) {
    DrawTool.drawSegmentations(source, segmentations, canvas, options);
  }
  drawPoseEstimations(source, poseEstimations, canvas, options = {}) {
    DrawTool.drawPoseEstimations(source, poseEstimations, canvas, options);
  }
  extractSegmentationEdgePoints(segmentation) {
    return DrawTool.extractSegmentationEdgePoints(segmentation);
  }
  extractSegmentationsEdgePoints(segmentations) {
    return DrawTool.extractSegmentationsEdgePoints(segmentations);
  }
  tensor(type, data, dims) {
    return new ort.Tensor(type, data, dims);
  }
  async getWebGpuDevice() {
    var _a;
    const device = (_a = ort.env.webgpu) == null ? void 0 : _a.device;
    if (!device) {
      throw new Error("WebGPU device is not initialized by ONNX Runtime Web.");
    }
    return device;
  }
  tensorFromGpuBuffer(gpuBuffer, dims, dispose) {
    return ort.Tensor.fromGpuBuffer(gpuBuffer, {
      dataType: "float32",
      dims,
      dispose
    });
  }
  async dispose() {
    if (!this.session) {
      return;
    }
    await this.session.release();
    this.session = null;
    this._onnxModel = null;
    this._handler = null;
    this.preprocessCanvas = null;
    this.preprocessContext = null;
    this.preprocessTensorData = null;
    this.preprocessTensorSize = 0;
  }
  createSessionOptions() {
    var _a, _b, _c;
    return {
      graphOptimizationLevel: "all",
      ...this.options.sessionOptions,
      executionProviders: (_c = (_a = this.options.sessionOptions) == null ? void 0 : _a.executionProviders) != null ? _c : [...(_b = this.options.executionProviders) != null ? _b : DEFAULT_EXECUTION_PROVIDERS]
    };
  }
  createSession(model) {
    return createOrtInferenceSession(model, this.createSessionOptions());
  }
  ensureSession() {
    if (!this.session) {
      throw new Error("Yolo model is not loaded. Call load() first or pass model to Yolo.create().");
    }
    return this.session;
  }
  getInputShape() {
    const shape = Object.values(this.onnxModel.inputShapes)[0];
    if (!shape || shape.length !== 4) {
      throw new Error(`Unsupported YOLO input shape: ${JSON.stringify(shape)}`);
    }
    return [shape[0], shape[1], shape[2], shape[3]];
  }
  getSourceRect(img, roi) {
    const { width, height } = this.getImageSourceSize(img);
    if (!roi) {
      return { x: 0, y: 0, width, height };
    }
    const left = this.clamp(Math.trunc(roi.left), 0, width - 1);
    const top = this.clamp(Math.trunc(roi.top), 0, height - 1);
    const right = this.clamp(Math.trunc(roi.right), left + 1, width);
    const bottom = this.clamp(Math.trunc(roi.bottom), top + 1, height);
    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top
    };
  }
  calculateProportionalResize(sourceWidth, sourceHeight, modelWidth, modelHeight) {
    if (sourceWidth < modelWidth && sourceHeight < modelHeight) {
      return {
        drawWidth: sourceWidth,
        drawHeight: sourceHeight,
        xPad: (modelWidth - sourceWidth) * 0.5,
        yPad: (modelHeight - sourceHeight) * 0.5,
        gain: 1
      };
    }
    const ratio = Math.min(modelWidth / sourceWidth, modelHeight / sourceHeight);
    const drawWidth = sourceWidth * ratio;
    const drawHeight = sourceHeight * ratio;
    return {
      drawWidth,
      drawHeight,
      xPad: (modelWidth - drawWidth) * 0.5,
      yPad: (modelHeight - drawHeight) * 0.5,
      gain: Math.max(sourceWidth / modelWidth, sourceHeight / modelHeight)
    };
  }
  clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }
  getPreprocessContext(width, height) {
    if (!this.preprocessCanvas) {
      this.preprocessCanvas = document.createElement("canvas");
    }
    if (this.preprocessCanvas.width !== width) {
      this.preprocessCanvas.width = width;
    }
    if (this.preprocessCanvas.height !== height) {
      this.preprocessCanvas.height = height;
    }
    if (!this.preprocessContext) {
      this.preprocessContext = this.preprocessCanvas.getContext("2d", { willReadFrequently: true });
    }
    if (!this.preprocessContext) {
      throw new Error("Canvas 2D context is not available.");
    }
    return this.preprocessContext;
  }
  getPreprocessTensorData(size) {
    if (!this.preprocessTensorData || this.preprocessTensorSize !== size) {
      this.preprocessTensorData = new Float32Array(size);
      this.preprocessTensorSize = size;
    }
    return this.preprocessTensorData;
  }
  getImageSourceSize(img) {
    if (img instanceof HTMLImageElement) {
      return {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height
      };
    }
    if (img instanceof HTMLVideoElement) {
      return {
        width: img.videoWidth || img.width,
        height: img.videoHeight || img.height
      };
    }
    if ("displayWidth" in img && "displayHeight" in img) {
      return {
        width: img.displayWidth || img.codedWidth,
        height: img.displayHeight || img.codedHeight
      };
    }
    if (img instanceof SVGImageElement) {
      const width = img.width.baseVal.value || img.getBoundingClientRect().width;
      const height = img.height.baseVal.value || img.getBoundingClientRect().height;
      return { width, height };
    }
    return {
      width: img.width,
      height: img.height
    };
  }
  ensureHandler() {
    if (!this._handler) {
      throw new Error("YOLO handler is not initialized. Call load() first or pass model to Yolo.create().");
    }
    return this._handler;
  }
  requireModel() {
    if (!this.model) {
      throw new Error("Missing model source. Pass model in constructor options or load(model).");
    }
    return this.model;
  }
  hasWebGpuExecutionProvider() {
    var _a, _b, _c;
    const executionProviders = (_c = (_b = (_a = this.options.sessionOptions) == null ? void 0 : _a.executionProviders) != null ? _b : this.options.executionProviders) != null ? _c : DEFAULT_EXECUTION_PROVIDERS;
    return executionProviders.some((executionProvider) => {
      if (typeof executionProvider === "string") {
        return executionProvider === "webgpu";
      }
      return (executionProvider == null ? void 0 : executionProvider.name) === "webgpu";
    });
  }
  isSupportedModel(modelVersion, modelType) {
    const allTasks = [
      "Classification",
      "ObjectDetection",
      "ObbDetection",
      "Segmentation",
      "PoseEstimation"
    ];
    const supportMap = {
      V5U: ["ObjectDetection"],
      V8: allTasks,
      V8E: ["Segmentation"],
      V9: ["ObjectDetection"],
      V10: ["ObjectDetection"],
      V11: allTasks,
      V11E: ["Segmentation"],
      V12: allTasks,
      V26: allTasks,
      RTDETR: ["ObjectDetection"],
      RFDETR: ["ObjectDetection", "Segmentation"],
      WORLDV2: ["ObjectDetection"]
    };
    return supportMap[modelVersion].includes(modelType);
  }
};

// src/handler/sam3/preprocess.ts
var SAM3_IMAGE_SIZE = 1008;
var SAM3_MASK_SIZE = 288;
var SAM3_TEXT_LENGTH = 32;
var SAM3_STD = 0.5;
var PIXEL_SCALE = 1 / (255 * SAM3_STD);
var PIXEL_BIAS = -0.5 / SAM3_STD;
var cpuCanvas = null;
var cpuContext = null;
var gpuCanvas = null;
var gpuContext = null;
var cpuTensorData = null;
function getImageSize(image) {
  if (image instanceof HTMLVideoElement) {
    return { width: image.videoWidth, height: image.videoHeight };
  }
  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth || image.width, height: image.naturalHeight || image.height };
  }
  if (image instanceof HTMLCanvasElement || image instanceof OffscreenCanvas) {
    return { width: image.width, height: image.height };
  }
  if ("displayWidth" in image && "displayHeight" in image) {
    return { width: Number(image.displayWidth), height: Number(image.displayHeight) };
  }
  const sized = image;
  return { width: sized.width, height: sized.height };
}
function enableHighQualitySmoothing(context) {
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
}
function getCachedCanvas(imageSize, willReadFrequently) {
  const existingCanvas = willReadFrequently ? cpuCanvas : gpuCanvas;
  const existingContext = willReadFrequently ? cpuContext : gpuContext;
  if (existingCanvas && existingContext && existingCanvas.width === imageSize && existingCanvas.height === imageSize) {
    return { canvas: existingCanvas, context: existingContext };
  }
  const canvas = document.createElement("canvas");
  canvas.width = imageSize;
  canvas.height = imageSize;
  const context = canvas.getContext("2d", { willReadFrequently });
  if (!context) {
    throw new Error("Failed to create a 2D canvas context for SAM3 preprocessing.");
  }
  if (willReadFrequently) {
    cpuCanvas = canvas;
    cpuContext = context;
  } else {
    gpuCanvas = canvas;
    gpuContext = context;
  }
  return { canvas, context };
}
function renderSam3ImageToCanvas(image, imageSize = SAM3_IMAGE_SIZE, willReadFrequently = true) {
  const { width: sourceWidth, height: sourceHeight } = getImageSize(image);
  const { canvas, context } = getCachedCanvas(imageSize, willReadFrequently);
  drawImageHighQuality(context, image, sourceWidth, sourceHeight, imageSize, imageSize);
  return { canvas, context, sourceWidth, sourceHeight };
}
function drawImageHighQuality(context, image, sourceWidth, sourceHeight, destWidth, destHeight) {
  enableHighQualitySmoothing(context);
  let src = image;
  let width = sourceWidth;
  let height = sourceHeight;
  while (width > destWidth * 2 || height > destHeight * 2) {
    const nextWidth = Math.max(destWidth, Math.floor(width / 2));
    const nextHeight = Math.max(destHeight, Math.floor(height / 2));
    const tmp = document.createElement("canvas");
    tmp.width = nextWidth;
    tmp.height = nextHeight;
    const tmpContext = tmp.getContext("2d");
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
function preprocessSam3Image(image, imageSize = SAM3_IMAGE_SIZE) {
  const { context, sourceWidth, sourceHeight } = renderSam3ImageToCanvas(image, imageSize, true);
  const pixels = context.getImageData(0, 0, imageSize, imageSize).data;
  const plane = imageSize * imageSize;
  const dataLength = 3 * plane;
  const data = cpuTensorData && cpuTensorData.length === dataLength ? cpuTensorData : new Float32Array(dataLength);
  cpuTensorData = data;
  for (let pixel = 0, offset = 0; pixel < plane; pixel += 1, offset += 4) {
    data[pixel] = pixels[offset] * PIXEL_SCALE + PIXEL_BIAS;
    data[plane + pixel] = pixels[offset + 1] * PIXEL_SCALE + PIXEL_BIAS;
    data[plane * 2 + pixel] = pixels[offset + 2] * PIXEL_SCALE + PIXEL_BIAS;
  }
  return { data, width: imageSize, height: imageSize, sourceWidth, sourceHeight };
}
function clamp2(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
function sigmoid2(value) {
  return 1 / (1 + Math.exp(-value));
}
function splitTextPrompts(prompt) {
  const trimmed = prompt.trim();
  if (!trimmed) {
    return [];
  }
  const parts = trimmed.split(/[,，]+/).map((part) => part.trim()).filter(Boolean);
  return parts.length > 0 ? parts : [trimmed];
}
function composeTextQuery(className, description) {
  const extra = description == null ? void 0 : description.trim();
  return extra ? `${className}, ${extra}` : className;
}
function rectCenter(box) {
  return { x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2 };
}
function packPcsGeometricPrompts(boxes, points) {
  const lastBox = boxes[boxes.length - 1];
  const packedBox = lastBox ? { box: lastBox.box, label: lastBox.label, pad: false } : { box: { left: 0, top: 0, right: 1, bottom: 1 }, label: true, pad: true };
  const lastPoint = points[points.length - 1];
  let packedPoint;
  if (lastPoint) {
    packedPoint = { point: lastPoint.point, label: lastPoint.label > 0 ? 1 : 0, pad: false };
  } else {
    const extraBox = boxes.length >= 2 ? boxes[boxes.length - 2] : void 0;
    packedPoint = extraBox ? { point: rectCenter(extraBox.box), label: extraBox.label ? 1 : 0, pad: false } : { point: { x: 0, y: 0 }, label: 1, pad: true };
  }
  return { boxes: [packedBox], points: [packedPoint] };
}
function rectToCxcywh(box, sourceWidth, sourceHeight) {
  const width = Math.max(box.right - box.left, 1);
  const height = Math.max(box.bottom - box.top, 1);
  return [
    clamp2((box.left + box.right) / 2 / sourceWidth, 0, 1),
    clamp2((box.top + box.bottom) / 2 / sourceHeight, 0, 1),
    clamp2(width / sourceWidth, 0, 1),
    clamp2(height / sourceHeight, 0, 1)
  ];
}
function cxcywhToRect(cx, cy, width, height, sourceWidth, sourceHeight) {
  const left = (cx - width / 2) * sourceWidth;
  const top = (cy - height / 2) * sourceHeight;
  const right = (cx + width / 2) * sourceWidth;
  const bottom = (cy + height / 2) * sourceHeight;
  return {
    left: clamp2(left, 0, sourceWidth),
    top: clamp2(top, 0, sourceHeight),
    right: clamp2(right, 0, sourceWidth),
    bottom: clamp2(bottom, 0, sourceHeight)
  };
}
function pointToNormalized(point, sourceWidth, sourceHeight) {
  return [clamp2(point.x / sourceWidth, 0, 1), clamp2(point.y / sourceHeight, 0, 1)];
}
function pointToModel(point, sourceWidth, sourceHeight, imageSize = SAM3_IMAGE_SIZE) {
  return [
    point.x / sourceWidth * imageSize,
    point.y / sourceHeight * imageSize
  ];
}
function boxToModelCorners(box, sourceWidth, sourceHeight, imageSize = SAM3_IMAGE_SIZE) {
  return [
    pointToModel({ x: box.left, y: box.top }, sourceWidth, sourceHeight, imageSize),
    pointToModel({ x: box.right, y: box.bottom }, sourceWidth, sourceHeight, imageSize)
  ];
}

// src/handler/sam3/postprocess.ts
var SAM3_MAX_OUTPUT_MASK_SIDE = 576;
function packBinaryMask(mask, width, height, threshold = 0.5) {
  var _a;
  const total = width * height;
  const packed = new Uint8Array(Math.ceil(total / 8));
  for (let index = 0; index < total; index += 1) {
    if (((_a = mask[index]) != null ? _a : 0) > threshold) {
      packed[index >> 3] |= 1 << (index & 7);
    }
  }
  return packed;
}
function bilinearResizeRegion(source, sourceWidth, sourceHeight, srcLeft, srcTop, srcRight, srcBottom, targetWidth, targetHeight) {
  var _a, _b, _c, _d;
  const target = new Float32Array(targetWidth * targetHeight);
  const regionWidth = Math.max(srcRight - srcLeft, 1e-6);
  const regionHeight = Math.max(srcBottom - srcTop, 1e-6);
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = srcTop + (y + 0.5) / targetHeight * regionHeight - 0.5;
    const y0 = clamp2(Math.floor(sourceY), 0, sourceHeight - 1);
    const y1 = y0 < sourceHeight - 1 ? y0 + 1 : y0;
    const wy = sourceY - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = srcLeft + (x + 0.5) / targetWidth * regionWidth - 0.5;
      const x0 = clamp2(Math.floor(sourceX), 0, sourceWidth - 1);
      const x1 = x0 < sourceWidth - 1 ? x0 + 1 : x0;
      const wx = sourceX - x0;
      const v00 = (_a = source[y0 * sourceWidth + x0]) != null ? _a : 0;
      const v01 = (_b = source[y0 * sourceWidth + x1]) != null ? _b : 0;
      const v10 = (_c = source[y1 * sourceWidth + x0]) != null ? _c : 0;
      const v11 = (_d = source[y1 * sourceWidth + x1]) != null ? _d : 0;
      target[y * targetWidth + x] = v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy;
    }
  }
  return target;
}
function bilinearResize(source, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  return bilinearResizeRegion(source, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight, targetWidth, targetHeight);
}
function maskBounds(mask, width, height, threshold = 0.5) {
  var _a;
  let left = width;
  let top = height;
  let right = 0;
  let bottom = 0;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      if (((_a = mask[row + x]) != null ? _a : 0) > threshold) {
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
function integerRect(rect, sourceWidth, sourceHeight) {
  const left = clamp2(Math.floor(rect.left), 0, sourceWidth);
  const top = clamp2(Math.floor(rect.top), 0, sourceHeight);
  const right = clamp2(Math.ceil(rect.right), left, sourceWidth);
  const bottom = clamp2(Math.ceil(rect.bottom), top, sourceHeight);
  return {
    left,
    top,
    right: right > left ? right : Math.min(sourceWidth, left + 1),
    bottom: bottom > top ? bottom : Math.min(sourceHeight, top + 1)
  };
}
function logitThreshold(pixelThreshold, applySigmoid) {
  if (!applySigmoid) {
    return pixelThreshold;
  }
  const probability = clamp2(pixelThreshold, 1e-6, 1 - 1e-6);
  return Math.log(probability / (1 - probability));
}
function sourceRectToMaskRect(box, sourceWidth, sourceHeight, maskWidth, maskHeight) {
  return {
    left: box.left / sourceWidth * maskWidth,
    top: box.top / sourceHeight * maskHeight,
    right: box.right / sourceWidth * maskWidth,
    bottom: box.bottom / sourceHeight * maskHeight
  };
}
function maskRectToSourceRect(box, maskWidth, maskHeight, sourceWidth, sourceHeight) {
  return {
    left: box.left / maskWidth * sourceWidth,
    top: box.top / maskHeight * sourceHeight,
    right: box.right / maskWidth * sourceWidth,
    bottom: box.bottom / maskHeight * sourceHeight
  };
}
function outputMaskSize(boxWidth, boxHeight) {
  const scale = Math.min(1, SAM3_MAX_OUTPUT_MASK_SIDE / Math.max(boxWidth, boxHeight, 1));
  return {
    width: Math.max(1, Math.round(boxWidth * scale)),
    height: Math.max(1, Math.round(boxHeight * scale))
  };
}
function applySigmoidInPlace(values) {
  var _a;
  for (let index = 0; index < values.length; index += 1) {
    values[index] = sigmoid2((_a = values[index]) != null ? _a : 0);
  }
  return values;
}
function probabilitiesToSegmentation(probabilities, maskWidth, maskHeight, box, label, confidence, pixelThreshold) {
  const packed = packBinaryMask(probabilities, maskWidth, maskHeight, pixelThreshold);
  const local = new Segmentation({
    label: { index: 0, name: label },
    confidence,
    boundingBox: { left: 0, top: 0, right: maskWidth, bottom: maskHeight },
    bitPackedPixelMask: packed
  });
  const localEdges = DrawTool.extractSegmentationEdgePoints(local);
  const scaleX = (box.right - box.left) / maskWidth;
  const scaleY = (box.bottom - box.top) / maskHeight;
  const edges = localEdges.map((point) => ({
    x: box.left + point.x * scaleX,
    y: box.top + point.y * scaleY
  }));
  return new Segmentation({
    label: { index: 0, name: label },
    confidence,
    boundingBox: box,
    bitPackedPixelMask: packed,
    segmentationEdgePoints: edges,
    pixelMaskWidth: maskWidth,
    pixelMaskHeight: maskHeight
  });
}
function logitsToSegmentation(logits, maskWidth, maskHeight, sourceWidth, sourceHeight, label, confidence, pixelThreshold = 0.5, applySigmoid = true, boundingBox) {
  const sourceBox = integerRect(
    boundingBox != null ? boundingBox : maskRectToSourceRect(
      maskBounds(logits, maskWidth, maskHeight, logitThreshold(pixelThreshold, applySigmoid)),
      maskWidth,
      maskHeight,
      sourceWidth,
      sourceHeight
    ),
    sourceWidth,
    sourceHeight
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
    output.height
  );
  const probabilities = applySigmoid ? applySigmoidInPlace(resized) : resized;
  return probabilitiesToSegmentation(
    probabilities,
    output.width,
    output.height,
    sourceBox,
    label,
    confidence,
    pixelThreshold
  );
}
function decodePcsOutputs(predMasks, predBoxes, predLogits, presenceLogits, maskShape, boxShape, sourceWidth, sourceHeight, text, threshold, pixelThreshold) {
  var _a, _b, _c, _d, _e, _f, _g, _h, _i;
  const queryCount = predLogits.length;
  const maskHeight = (_a = maskShape[maskShape.length - 2]) != null ? _a : SAM3_MASK_SIZE;
  const maskWidth = (_b = maskShape[maskShape.length - 1]) != null ? _b : SAM3_MASK_SIZE;
  const boxStride = (_c = boxShape[boxShape.length - 1]) != null ? _c : 4;
  const presence = sigmoid2((_d = presenceLogits[0]) != null ? _d : 0);
  const results = [];
  for (let query = 0; query < queryCount; query += 1) {
    const score = sigmoid2((_e = predLogits[query]) != null ? _e : 0) * presence;
    if (score <= threshold) {
      continue;
    }
    const maskOffset = query * maskHeight * maskWidth;
    const mask = predMasks.subarray(maskOffset, maskOffset + maskHeight * maskWidth);
    const boxOffset = query * boxStride;
    const boundingBox = boxStride >= 4 ? cxcywhToRect(
      (_f = predBoxes[boxOffset]) != null ? _f : 0,
      (_g = predBoxes[boxOffset + 1]) != null ? _g : 0,
      (_h = predBoxes[boxOffset + 2]) != null ? _h : 0,
      (_i = predBoxes[boxOffset + 3]) != null ? _i : 0,
      sourceWidth,
      sourceHeight
    ) : void 0;
    results.push(
      logitsToSegmentation(
        mask,
        maskWidth,
        maskHeight,
        sourceWidth,
        sourceHeight,
        text || "object",
        score,
        pixelThreshold,
        true,
        boundingBox
      )
    );
  }
  return results;
}
function copyLowResMask(data, index, width, height) {
  const size = width * height;
  return data.slice(index * size, (index + 1) * size);
}

// src/handler/sam3/webgpu-preprocess.ts
var PREPROCESS_SHADER = `
struct Params {
  size: vec4<f32>,
  scale: vec4<f32>,
  bias: vec4<f32>,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(params.size.x);
  let height = u32(params.size.y);

  if (id.x >= width || id.y >= height) {
    return;
  }

  let rgba = textureLoad(inputTexture, vec2<i32>(i32(id.x), i32(id.y)), 0);
  let pixel = id.y * width + id.x;
  let planeSize = width * height;

  outputTensor[pixel] = rgba.r * params.scale.x + params.bias.x;
  outputTensor[planeSize + pixel] = rgba.g * params.scale.y + params.bias.y;
  outputTensor[planeSize * 2u + pixel] = rgba.b * params.scale.z + params.bias.z;
}
`;
async function getOrtWebGpuDevice() {
  const webgpu = ort.env.webgpu;
  const device = await (webgpu == null ? void 0 : webgpu.device);
  if (!device) {
    throw new Error("ONNX Runtime WebGPU device is not initialized.");
  }
  return device;
}
async function waitForWebGpuOutputs(_result) {
  var _a, _b;
  const device = await ((_a = ort.env.webgpu) == null ? void 0 : _a.device);
  if (typeof ((_b = device == null ? void 0 : device.queue) == null ? void 0 : _b.onSubmittedWorkDone) === "function") {
    await device.queue.onSubmittedWorkDone();
  }
}
var Sam3WebGpuPreprocessor = class {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.texture = null;
    this.outputBuffer = null;
    this.paramsBuffer = null;
    this.size = 0;
  }
  async process(image, imageSize = SAM3_IMAGE_SIZE) {
    const { source, sourceWidth, sourceHeight, close } = await rasterizeSam3Square(image, imageSize);
    const device = await this.ensureResources(imageSize);
    try {
      device.queue.copyExternalImageToTexture({ source }, { texture: this.texture }, { width: imageSize, height: imageSize });
    } finally {
      close == null ? void 0 : close();
    }
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: this.outputBuffer } },
        { binding: 2, resource: { buffer: this.paramsBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(imageSize / 16), Math.ceil(imageSize / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);
    return {
      data: new Float32Array(0),
      width: imageSize,
      height: imageSize,
      sourceWidth,
      sourceHeight,
      tensor: ort.Tensor.fromGpuBuffer(this.outputBuffer, {
        dataType: "float32",
        dims: [1, 3, imageSize, imageSize]
      })
    };
  }
  dispose() {
    var _a, _b, _c;
    (_a = this.texture) == null ? void 0 : _a.destroy();
    (_b = this.outputBuffer) == null ? void 0 : _b.destroy();
    (_c = this.paramsBuffer) == null ? void 0 : _c.destroy();
    this.texture = null;
    this.outputBuffer = null;
    this.paramsBuffer = null;
    this.pipeline = null;
    this.device = null;
    this.size = 0;
  }
  async ensureResources(imageSize) {
    const device = await getOrtWebGpuDevice();
    const usage = globalThis.GPUBufferUsage;
    const textureUsage = globalThis.GPUTextureUsage;
    if (!usage || !textureUsage) {
      throw new Error("WebGPU buffer usage flags are not available.");
    }
    if (this.device !== device || this.size !== imageSize || !this.pipeline || !this.texture || !this.outputBuffer || !this.paramsBuffer) {
      this.dispose();
      this.device = device;
      this.size = imageSize;
      this.texture = device.createTexture({
        size: [imageSize, imageSize, 1],
        format: "rgba8unorm",
        usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT
      });
      this.outputBuffer = device.createBuffer({
        size: 3 * imageSize * imageSize * Float32Array.BYTES_PER_ELEMENT,
        usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST
      });
      this.paramsBuffer = device.createBuffer({
        size: 48,
        usage: usage.UNIFORM | usage.COPY_DST
      });
      this.pipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: PREPROCESS_SHADER }),
          entryPoint: "main"
        }
      });
      const scale = 1 / SAM3_STD;
      const bias = -0.5 / SAM3_STD;
      device.queue.writeBuffer(
        this.paramsBuffer,
        0,
        new Float32Array([imageSize, imageSize, 0, 0, scale, scale, scale, 0, bias, bias, bias, 0])
      );
    }
    return device;
  }
};
async function rasterizeSam3Square(image, imageSize) {
  const { width: sourceWidth, height: sourceHeight } = getImageSize(image);
  try {
    const bitmap = await createImageBitmap(image, {
      resizeWidth: imageSize,
      resizeHeight: imageSize,
      resizeQuality: "high"
    });
    return { source: bitmap, sourceWidth, sourceHeight, close: () => bitmap.close() };
  } catch (e) {
    const { canvas } = renderSam3ImageToCanvas(image, imageSize, false);
    return { source: canvas, sourceWidth, sourceHeight };
  }
}

// src/handler/sam3/sam3-handler.ts
var TEXT_OUTPUT_NAMES = ["language_mask", "language_features"];
var TEXT_CACHE_LIMIT = 32;
function asFloat32(data) {
  if (data instanceof Float32Array) {
    return data;
  }
  return Float32Array.from(data);
}
function copyFloat32(data) {
  return new Float32Array(data);
}
function pickFeeds(session, candidates) {
  const feeds = {};
  for (const name of session.inputNames) {
    const tensor = candidates[name];
    if (!tensor) {
      throw new Error(`SAM3 session is missing feed "${name}". Available: ${Object.keys(candidates).join(", ")}`);
    }
    feeds[name] = tensor;
  }
  return feeds;
}
function pickOutput(result, name) {
  const tensor = result[name];
  if (!tensor) {
    throw new Error(`SAM3 output "${name}" is missing. Available: ${Object.keys(result).join(", ")}`);
  }
  return tensor;
}
function optionalOutput(result, name) {
  return result[name];
}
function tensorFloat(data, dims) {
  return new ort.Tensor("float32", data, [...dims]);
}
function tensorInt64(values, dims) {
  const data = BigInt64Array.from(Array.from(values, (value) => BigInt(value)));
  return new ort.Tensor("int64", data, [...dims]);
}
function tensorInt32(values, dims) {
  return new ort.Tensor("int32", values, [...dims]);
}
function tensorBool(values, dims) {
  return new ort.Tensor("bool", values, [...dims]);
}
function selectPvsMaskIndices(maskCount, ious, wantMulti) {
  if (maskCount >= 4) {
    return wantMulti ? [1, 2, 3] : [0];
  }
  if (maskCount <= 1) {
    return [0];
  }
  return wantMulti ? Array.from({ length: maskCount }, (_, index) => index) : [argmax(ious)];
}
function ensureFloat32(current, length) {
  return current && current.length >= length ? current : new Float32Array(length);
}
function ensureInt32(current, length) {
  return current && current.length >= length ? current : new Int32Array(length);
}
function fillMaskInput(target, maskInput, maskSize) {
  const needed = maskSize * maskSize;
  const maskData = ensureFloat32(target, needed);
  maskData.fill(0, 0, needed);
  if (!maskInput) {
    return maskData;
  }
  if (maskInput.width === maskSize && maskInput.height === maskSize) {
    maskData.set(maskInput.logits.subarray(0, needed));
    return maskData;
  }
  maskData.set(bilinearResize(maskInput.logits, maskInput.width, maskInput.height, maskSize, maskSize));
  return maskData;
}
function cloneCpuTensor(tensor) {
  const dims = [...tensor.dims];
  if (tensor.type === "bool") {
    return new ort.Tensor("bool", Uint8Array.from(tensor.data), dims);
  }
  if (tensor.type === "int64") {
    return new ort.Tensor("int64", BigInt64Array.from(tensor.data), dims);
  }
  if (tensor.type === "int32") {
    return new ort.Tensor("int32", Int32Array.from(tensor.data), dims);
  }
  return new ort.Tensor("float32", copyFloat32(asFloat32(tensor.data)), dims);
}
function sliceAlongAxis(data, dims, axis, index) {
  var _a;
  const inner = dims.slice(axis + 1).reduce((product, dim) => product * dim, 1);
  const outer = dims.slice(0, axis).reduce((product, dim) => product * dim, 1);
  const axisSize = (_a = dims[axis]) != null ? _a : 1;
  const out = data instanceof Uint8Array ? new Uint8Array(outer * inner) : new Float32Array(outer * inner);
  for (let prefix = 0; prefix < outer; prefix += 1) {
    const source = (prefix * axisSize + index) * inner;
    out.set(data.subarray(source, source + inner), prefix * inner);
  }
  return {
    data: out,
    dims: dims.map((dim, dimIndex) => dimIndex === axis ? 1 : dim)
  };
}
function featureBatchAxis(dims, batch) {
  if (dims.length >= 3 && dims[1] === batch) {
    return 1;
  }
  if (dims[0] === batch) {
    return 0;
  }
  const axis = dims.indexOf(batch);
  if (axis < 0) {
    throw new Error(`Text encoder output shape [${dims.join(", ")}] has no batch=${batch} axis.`);
  }
  return axis;
}
async function runTextSession(session, feeds) {
  try {
    return await session.run(feeds, [...TEXT_OUTPUT_NAMES]);
  } catch (e) {
    return session.run(feeds);
  }
}
var Sam3Handler = class {
  constructor(visionSession, textSession, groundingSession, promptSession, tokenizer, options) {
    this.visionSession = visionSession;
    this.textSession = textSession;
    this.groundingSession = groundingSession;
    this.promptSession = promptSession;
    this.tokenizer = tokenizer;
    this.options = options;
    this.textEmbeddingCache = /* @__PURE__ */ new Map();
    this.pvsMaskScratch = null;
    this.pvsCoordsScratch = null;
    this.pvsLabelsScratch = null;
    this.pvsHasMaskScratch = new Float32Array(1);
    this.webGpuPreprocessor = options.webGpu ? new Sam3WebGpuPreprocessor() : null;
  }
  setConfidenceThreshold(threshold, state) {
    this.options.confidenceThreshold = threshold;
    return state ? this.options.confidenceThreshold : threshold;
  }
  async applyConfidenceThreshold(state) {
    var _a;
    if ((_a = state.lastPcsRaw) == null ? void 0 : _a.length) {
      const merged = [];
      for (const raw of state.lastPcsRaw) {
        merged.push(...this.decodePcsRaw(state, raw));
      }
      state.lastPcs = merged;
      return merged;
    }
    return this.forwardAllTextPrompts(state);
  }
  dispose() {
    var _a;
    (_a = this.webGpuPreprocessor) == null ? void 0 : _a.dispose();
    for (const embeddings of this.textEmbeddingCache.values()) {
      this.releaseTensors(embeddings);
    }
    this.textEmbeddingCache.clear();
  }
  async setImage(image, state) {
    var _a, _b, _c;
    if (state == null ? void 0 : state.vision) {
      this.releaseVision(state.vision);
    }
    const preprocessStarted = performance.now();
    const input = await this.prepareImage(image);
    const preprocessMs = performance.now() - preprocessStarted;
    let imageTensor = (_a = input.tensor) != null ? _a : tensorFloat(input.data, [1, 3, input.height, input.width]);
    let result;
    let runMs = 0;
    let fenceMs = 0;
    try {
      try {
        const runStarted = performance.now();
        result = await this.visionSession.run(pickFeeds(this.visionSession, { images: imageTensor }));
        runMs = performance.now() - runStarted;
      } catch (error) {
        if (!input.tensor) {
          throw error;
        }
        console.warn("[SAM3] GPU image tensor was rejected. Falling back to CPU input.", error);
        (_b = imageTensor.dispose) == null ? void 0 : _b.call(imageTensor);
        const cpu = preprocessSam3Image(image, this.options.imageSize);
        imageTensor = tensorFloat(cpu.data, [1, 3, cpu.height, cpu.width]);
        const runStarted = performance.now();
        result = await this.visionSession.run(pickFeeds(this.visionSession, { images: imageTensor }));
        runMs = performance.now() - runStarted;
      }
      if (this.options.webGpu) {
        const fenceStarted = performance.now();
        await waitForWebGpuOutputs(result);
        fenceMs = performance.now() - fenceStarted;
      }
    } finally {
      (_c = imageTensor.dispose) == null ? void 0 : _c.call(imageTensor);
    }
    console.info(
      `[SAM3] encode preprocess=${preprocessMs.toFixed(0)}ms run=${runMs.toFixed(0)}ms fence=${fenceMs.toFixed(0)}ms total=${(preprocessMs + runMs + fenceMs).toFixed(0)}ms`
    );
    return {
      sourceWidth: input.sourceWidth,
      sourceHeight: input.sourceHeight,
      vision: {
        detFpn0: pickOutput(result, "det_fpn_0"),
        detFpn1: pickOutput(result, "det_fpn_1"),
        detFpn2: pickOutput(result, "det_fpn_2"),
        detPos0: optionalOutput(result, "det_pos_0"),
        detPos1: optionalOutput(result, "det_pos_1"),
        detPos2: optionalOutput(result, "det_pos_2"),
        pvsHighRes0: pickOutput(result, "pvs_high_res_0"),
        pvsHighRes1: pickOutput(result, "pvs_high_res_1"),
        pvsImageEmbed: pickOutput(result, "pvs_image_embed")
      },
      boxes: [],
      points: [],
      pvsPoints: [],
      pvsBox: null,
      pvsMaskInput: null
    };
  }
  resetPrompts(state) {
    state.text = void 0;
    state.texts = void 0;
    state.textDescription = void 0;
    state.textEmbeddings = void 0;
    state.boxes = [];
    state.points = [];
    state.lastPcs = void 0;
    state.lastPcsRaw = void 0;
    return state;
  }
  resetVisualPrompts(state) {
    state.pvsPoints = [];
    state.pvsBox = null;
    state.pvsMaskInput = null;
    state.lastPvs = void 0;
    return state;
  }
  async setTextPrompt(prompt, state, description) {
    this.ensureImage(state);
    const original = (prompt || "visual").trim() || "visual";
    state.text = original;
    state.textDescription = (description == null ? void 0 : description.trim()) || void 0;
    state.texts = splitTextPrompts(original);
    if (state.texts.length === 0) {
      state.texts = [state.text];
    }
    return this.forwardAllTextPrompts(state);
  }
  async addGeometricPrompt(box, label, state) {
    this.ensureImage(state);
    state.boxes.push({ box, label });
    try {
      return await this.forwardAllTextPrompts(state);
    } catch (error) {
      state.boxes.pop();
      throw error;
    }
  }
  async addGeometricPoint(point, label, state) {
    this.ensureImage(state);
    state.points.push({ point, label: label ? 1 : 0 });
    try {
      return await this.forwardAllTextPrompts(state);
    } catch (error) {
      state.points.pop();
      throw error;
    }
  }
  async removeGeometricPrompt(index, state) {
    if (index >= 0 && index < state.boxes.length) {
      state.boxes.splice(index, 1);
    }
    return this.forwardAllTextPrompts(state);
  }
  async removeGeometricPoint(index, state) {
    if (index >= 0 && index < state.points.length) {
      state.points.splice(index, 1);
    }
    return this.forwardAllTextPrompts(state);
  }
  async predictConcept(prompt, state) {
    if (prompt.text !== void 0) {
      const original = (prompt.text || "visual").trim() || "visual";
      state.text = original;
      state.texts = splitTextPrompts(original);
      if (state.texts.length === 0) {
        state.texts = [state.text];
      }
    }
    if (prompt.description !== void 0) {
      state.textDescription = prompt.description.trim() || void 0;
    }
    if (prompt.boxes) {
      state.boxes = [...prompt.boxes];
    }
    if (prompt.points) {
      state.points = [...prompt.points];
    }
    return this.forwardAllTextPrompts(state);
  }
  async predictVisual(prompt, state) {
    var _a, _b, _c, _d, _e, _f, _g;
    this.ensureImage(state);
    if (!this.promptSession) {
      throw new Error("SAM3 prompt decoder is not loaded. Pass promptDecoder when creating Sam3.");
    }
    const points = [...(_a = prompt.points) != null ? _a : state.pvsPoints];
    const box = prompt.box === void 0 ? state.pvsBox : prompt.box;
    const maskInput = prompt.maskInput === void 0 ? state.pvsMaskInput : prompt.maskInput;
    const concatPoints = [];
    if (box) {
      const corners = boxToModelCorners(box, state.sourceWidth, state.sourceHeight, this.options.imageSize);
      concatPoints.push({ x: corners[0][0], y: corners[0][1], label: 2 }, { x: corners[1][0], y: corners[1][1], label: 3 });
    }
    for (const item of points) {
      const [x, y] = pointToModel(item.point, state.sourceWidth, state.sourceHeight, this.options.imageSize);
      concatPoints.push({ x, y, label: item.label });
    }
    if (concatPoints.length === 0) {
      concatPoints.push({ x: 0, y: 0, label: -1 });
    }
    const coordLength = concatPoints.length * 2;
    this.pvsCoordsScratch = ensureFloat32(this.pvsCoordsScratch, coordLength);
    this.pvsLabelsScratch = ensureInt32(this.pvsLabelsScratch, concatPoints.length);
    const coords = this.pvsCoordsScratch.subarray(0, coordLength);
    const labels = this.pvsLabelsScratch.subarray(0, concatPoints.length);
    for (let index = 0; index < concatPoints.length; index += 1) {
      coords[index * 2] = concatPoints[index].x;
      coords[index * 2 + 1] = concatPoints[index].y;
      labels[index] = concatPoints[index].label;
    }
    const maskSize = this.options.maskSize;
    this.pvsHasMaskScratch[0] = maskInput ? 1 : 0;
    this.pvsMaskScratch = fillMaskInput(this.pvsMaskScratch, maskInput, maskSize);
    const maskData = this.pvsMaskScratch.subarray(0, maskSize * maskSize);
    const result = await this.promptSession.run(
      pickFeeds(this.promptSession, {
        image_embed: state.vision.pvsImageEmbed,
        high_res_0: state.vision.pvsHighRes0,
        high_res_1: state.vision.pvsHighRes1,
        point_coords: tensorFloat(coords, [1, concatPoints.length, 2]),
        point_labels: tensorInt32(labels, [1, concatPoints.length]),
        mask_input: tensorFloat(maskData, [1, 1, maskSize, maskSize]),
        has_mask_input: tensorFloat(this.pvsHasMaskScratch, [1])
      })
    );
    const lowRes = pickOutput(result, "low_res_masks");
    const ious = asFloat32(pickOutput(result, "iou_predictions").data);
    const objectScoreLogits = result.object_score_logits ? asFloat32(result.object_score_logits.data) : [];
    const objectScore = objectScoreLogits.length > 0 ? 1 / (1 + Math.exp(-((_b = objectScoreLogits[0]) != null ? _b : 0))) : 1;
    const lowResData = asFloat32(lowRes.data);
    const dims = lowRes.dims;
    const maskCount = dims.length >= 4 ? dims[1] : 1;
    const height = (_c = dims[dims.length - 2]) != null ? _c : maskSize;
    const width = (_d = dims[dims.length - 1]) != null ? _d : maskSize;
    const clickCount = concatPoints.filter((item) => item.label >= 0 && item.label <= 1).length;
    const wantMulti = (_e = prompt.multimaskOutput) != null ? _e : clickCount <= 1 && !box && !maskInput;
    const selected = selectPvsMaskIndices(maskCount, ious, wantMulti);
    const masks = [];
    const lowResMasks = [];
    const selectedIous = [];
    const selectedScores = [];
    for (const index of selected) {
      const slice = copyLowResMask(lowResData, index, width, height);
      const predictedIou = (_f = ious[index]) != null ? _f : 0;
      lowResMasks.push(slice);
      selectedIous.push(predictedIou);
      selectedScores.push(objectScore);
      masks.push(
        logitsToSegmentation(
          slice,
          width,
          height,
          state.sourceWidth,
          state.sourceHeight,
          "object",
          clamp2(predictedIou, 0, 1),
          0,
          false
        )
      );
    }
    const pvsResult = {
      masks,
      lowResMasks,
      ious: selectedIous,
      objectScores: selectedScores,
      maskWidth: width,
      maskHeight: height
    };
    if (prompt.persist !== false) {
      state.pvsPoints = points;
      state.pvsBox = box;
      state.lastPvs = pvsResult;
      const bestIndex = argmax(selectedIous);
      state.pvsMaskInput = {
        logits: (_g = lowResMasks[bestIndex]) != null ? _g : lowResMasks[0],
        width,
        height
      };
    }
    return pvsResult;
  }
  selectPvsCandidate(index, state) {
    var _a, _b, _c, _d, _e;
    const last = state.lastPvs;
    const mask = last == null ? void 0 : last.masks[index];
    const logits = last == null ? void 0 : last.lowResMasks[index];
    if (!last || !mask || !logits) {
      throw new Error(`PVS candidate ${index} is not available. Run predictVisual() first.`);
    }
    const width = (_a = last.maskWidth) != null ? _a : this.options.maskSize;
    const height = (_b = last.maskHeight) != null ? _b : this.options.maskSize;
    state.pvsMaskInput = { logits, width, height };
    return {
      masks: [mask],
      lowResMasks: [logits],
      ious: [(_c = last.ious[index]) != null ? _c : 0],
      objectScores: [(_e = (_d = last.objectScores[index]) != null ? _d : last.objectScores[0]) != null ? _e : 0],
      maskWidth: width,
      maskHeight: height
    };
  }
  async addPoint(point, label, state) {
    state.pvsPoints.push({ point, label });
    const promptCount = state.pvsPoints.length + (state.pvsBox ? 1 : 0);
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box: state.pvsBox,
        maskInput: promptCount <= 1 ? null : state.pvsMaskInput,
        multimaskOutput: promptCount <= 1
      },
      state
    );
  }
  async addBox(box, state) {
    state.pvsBox = box;
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box,
        maskInput: state.pvsPoints.length === 0 ? null : state.pvsMaskInput,
        multimaskOutput: false
      },
      state
    );
  }
  async addMask(mask, state) {
    state.pvsMaskInput = mask;
    return this.predictVisual(
      {
        points: state.pvsPoints,
        box: state.pvsBox,
        maskInput: mask,
        multimaskOutput: false
      },
      state
    );
  }
  async removePoint(index, state) {
    if (index >= 0 && index < state.pvsPoints.length) {
      state.pvsPoints.splice(index, 1);
    }
    state.pvsMaskInput = null;
    return this.predictVisual({ points: state.pvsPoints, box: state.pvsBox, maskInput: null }, state);
  }
  cacheTextEmbeddings(query, embeddings) {
    const previous = this.textEmbeddingCache.get(query);
    if (previous && previous !== embeddings) {
      this.releaseTensors(previous);
    }
    this.textEmbeddingCache.delete(query);
    this.textEmbeddingCache.set(query, embeddings);
    while (this.textEmbeddingCache.size > TEXT_CACHE_LIMIT) {
      const oldest = this.textEmbeddingCache.keys().next().value;
      if (oldest === void 0) {
        break;
      }
      const stale = this.textEmbeddingCache.get(oldest);
      this.textEmbeddingCache.delete(oldest);
      if (stale) {
        this.releaseTensors(stale);
      }
    }
  }
  async encodeTextUncached(query) {
    const ids = this.tokenizer.tokenize(query);
    const result = await runTextSession(
      this.textSession,
      pickFeeds(this.textSession, {
        input_ids: tensorInt64(ids, [1, ids.length])
      })
    );
    this.cacheTextEmbeddings(query, {
      languageMask: cloneCpuTensor(pickOutput(result, "language_mask")),
      languageFeatures: cloneCpuTensor(pickOutput(result, "language_features"))
    });
  }
  async encodeTextsBatched(queries) {
    var _a, _b, _c;
    const batch = queries.length;
    const length = this.tokenizer.contextLength;
    const ids = new BigInt64Array(batch * length);
    for (let index = 0; index < batch; index += 1) {
      const tokens = this.tokenizer.tokenize((_a = queries[index]) != null ? _a : "");
      for (let token = 0; token < length; token += 1) {
        ids[index * length + token] = BigInt((_b = tokens[token]) != null ? _b : 0);
      }
    }
    const result = await runTextSession(
      this.textSession,
      pickFeeds(this.textSession, {
        input_ids: new ort.Tensor("int64", ids, [batch, length])
      })
    );
    const maskTensor = pickOutput(result, "language_mask");
    const featureTensor = pickOutput(result, "language_features");
    const maskDims = maskTensor.dims;
    const featureDims = featureTensor.dims;
    const maskAxis = maskDims[0] === batch ? 0 : featureBatchAxis(maskDims, batch);
    const featureAxis = featureBatchAxis(featureDims, batch);
    const maskData = maskTensor.data instanceof Uint8Array ? maskTensor.data : Uint8Array.from(maskTensor.data);
    const featureData = asFloat32(featureTensor.data);
    for (let index = 0; index < batch; index += 1) {
      const maskSlice = sliceAlongAxis(maskData, maskDims, maskAxis, index);
      const featureSlice = sliceAlongAxis(featureData, featureDims, featureAxis, index);
      this.cacheTextEmbeddings((_c = queries[index]) != null ? _c : "", {
        languageMask: new ort.Tensor("bool", maskSlice.data, maskSlice.dims),
        languageFeatures: new ort.Tensor("float32", featureSlice.data, featureSlice.dims)
      });
    }
  }
  async ensureTextEmbeddings(queries) {
    var _a;
    const missing = [...new Set(queries)].filter((query) => !this.textEmbeddingCache.has(query));
    if (missing.length === 0) {
      return;
    }
    if (missing.length === 1) {
      await this.encodeTextUncached((_a = missing[0]) != null ? _a : "");
      return;
    }
    try {
      await this.encodeTextsBatched(missing);
    } catch (error) {
      console.warn("[SAM3] Batched text encode failed, falling back to sequential.", error);
      for (const query of missing) {
        if (!this.textEmbeddingCache.has(query)) {
          await this.encodeTextUncached(query);
        }
      }
    }
  }
  async forwardAllTextPrompts(state) {
    var _a, _b, _c, _d;
    this.ensureImage(state);
    const original = ((_a = state.text) == null ? void 0 : _a.trim()) || "visual";
    const texts = ((_b = state.texts) == null ? void 0 : _b.length) ? state.texts : splitTextPrompts(original);
    const queries = texts.map((text) => composeTextQuery(text, state.textDescription));
    await this.ensureTextEmbeddings(queries);
    const merged = [];
    const rawOutputs = [];
    for (let index = 0; index < texts.length; index += 1) {
      const embeddings = this.textEmbeddingCache.get((_c = queries[index]) != null ? _c : "");
      if (!embeddings) {
        throw new Error(`Text embeddings are missing for "${queries[index]}".`);
      }
      state.textEmbeddings = embeddings;
      const raw = await this.runGrounding(state, (_d = texts[index]) != null ? _d : original);
      rawOutputs.push(raw);
      merged.push(...this.decodePcsRaw(state, raw));
    }
    state.text = original;
    state.texts = texts;
    state.lastPcsRaw = rawOutputs;
    state.lastPcs = merged;
    return merged;
  }
  decodePcsRaw(state, raw) {
    return decodePcsOutputs(
      raw.predMasks,
      raw.predBoxes,
      raw.predLogits,
      raw.presenceLogits,
      raw.maskShape,
      raw.boxShape,
      state.sourceWidth,
      state.sourceHeight,
      raw.label,
      this.options.confidenceThreshold,
      this.options.pixelConfidence
    );
  }
  async runGrounding(state, label) {
    var _a, _b, _c;
    this.ensureImage(state);
    if (!state.textEmbeddings) {
      throw new Error("Text embeddings are missing. Call setTextPrompt() first.");
    }
    const packed = packPcsGeometricPrompts(state.boxes, state.points);
    const boxes = packed.boxes;
    const points = packed.points;
    const boxCoords = new Float32Array(boxes.length * 4);
    const boxLabels = new Int32Array(boxes.length);
    const boxPadMask = new Uint8Array(boxes.length);
    for (let index = 0; index < boxes.length; index += 1) {
      const cxcywh = rectToCxcywh(boxes[index].box, state.sourceWidth, state.sourceHeight);
      boxCoords[index * 4] = cxcywh[0];
      boxCoords[index * 4 + 1] = cxcywh[1];
      boxCoords[index * 4 + 2] = cxcywh[2];
      boxCoords[index * 4 + 3] = cxcywh[3];
      boxLabels[index] = boxes[index].label ? 1 : 0;
      boxPadMask[index] = boxes[index].pad ? 1 : 0;
    }
    const pointCoords = new Float32Array(points.length * 2);
    const pointLabels = new Int32Array(points.length);
    const pointPadMask = new Uint8Array(points.length);
    for (let index = 0; index < points.length; index += 1) {
      const xy = pointToNormalized(points[index].point, state.sourceWidth, state.sourceHeight);
      pointCoords[index * 2] = xy[0];
      pointCoords[index * 2 + 1] = xy[1];
      pointLabels[index] = points[index].label > 0 ? 1 : 0;
      pointPadMask[index] = points[index].pad ? 1 : 0;
    }
    let result;
    try {
      result = await this.groundingSession.run(
        pickFeeds(this.groundingSession, {
          det_fpn_0: state.vision.detFpn0,
          det_fpn_1: state.vision.detFpn1,
          det_fpn_2: state.vision.detFpn2,
          det_pos_0: state.vision.detPos0,
          det_pos_1: state.vision.detPos1,
          det_pos_2: state.vision.detPos2,
          language_features: state.textEmbeddings.languageFeatures,
          language_mask: state.textEmbeddings.languageMask,
          box_coords: tensorFloat(boxCoords, [boxes.length, 1, 4]),
          box_labels: tensorInt64(boxLabels, [boxes.length, 1]),
          box_pad_mask: tensorBool(boxPadMask, [1, boxes.length]),
          point_coords: tensorFloat(pointCoords, [points.length, 1, 2]),
          point_labels: tensorInt64(pointLabels, [points.length, 1]),
          point_pad_mask: tensorBool(pointPadMask, [1, points.length])
        })
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `SAM3 grounding decoder failed with ${state.boxes.length} box(es) and ${state.points.length} point(s): ${message}`
      );
    }
    const predMasks = pickOutput(result, "pred_masks");
    const predBoxes = pickOutput(result, "pred_boxes");
    return {
      predMasks: copyFloat32(asFloat32(predMasks.data)),
      predBoxes: copyFloat32(asFloat32(predBoxes.data)),
      predLogits: copyFloat32(asFloat32(pickOutput(result, "pred_logits").data)),
      presenceLogits: copyFloat32(asFloat32(pickOutput(result, "presence_logits").data)),
      maskShape: [...predMasks.dims],
      boxShape: [...predBoxes.dims],
      label: (_c = (_b = label != null ? label : (_a = state.texts) == null ? void 0 : _a[0]) != null ? _b : state.text) != null ? _c : "visual"
    };
  }
  async prepareImage(image) {
    if (this.webGpuPreprocessor) {
      try {
        return await this.webGpuPreprocessor.process(image, this.options.imageSize);
      } catch (error) {
        console.warn("[SAM3] WebGPU preprocessing failed. Falling back to CPU preprocessing.", error);
      }
    }
    return preprocessSam3Image(image, this.options.imageSize);
  }
  ensureImage(state) {
    if (!(state == null ? void 0 : state.vision)) {
      throw new Error("You must call setImage() before prompting SAM3.");
    }
  }
  releaseTensors(tensors) {
    var _a;
    for (const tensor of Object.values(tensors)) {
      (_a = tensor == null ? void 0 : tensor.dispose) == null ? void 0 : _a.call(tensor);
    }
  }
  releaseVision(vision) {
    this.releaseTensors(vision);
  }
};
function argmax(values) {
  var _a, _b;
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    if (((_a = values[index]) != null ? _a : Number.NEGATIVE_INFINITY) > bestValue) {
      bestValue = (_b = values[index]) != null ? _b : Number.NEGATIVE_INFINITY;
      bestIndex = index;
    }
  }
  return bestIndex;
}

// src/handler/sam3/clip-tokenizer.ts
var TOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+/gu;
function unescapeHtml(text) {
  return text.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code))).replace(/&#x([0-9a-fA-F]+);/g, (_, code) => String.fromCodePoint(parseInt(code, 16)));
}
function whitespaceClean(text) {
  return text.replace(/\s+/g, " ").trim();
}
function getPairs(word) {
  const pairs = /* @__PURE__ */ new Set();
  for (let index = 0; index < word.length - 1; index += 1) {
    pairs.add(`${word[index]} ${word[index + 1]}`);
  }
  return pairs;
}
function applyBpe(token, tables, cache) {
  var _a;
  const cached = cache.get(token);
  if (cached !== void 0) {
    return cached;
  }
  if (token.length === 0) {
    return token;
  }
  let word = token.slice(0, -1).split("").concat(`${token.slice(-1)}</w>`);
  let pairs = getPairs(word);
  if (pairs.size === 0) {
    const result2 = `${token}</w>`;
    cache.set(token, result2);
    return result2;
  }
  while (true) {
    let minRank = Number.POSITIVE_INFINITY;
    let bigram = null;
    for (const pair of pairs) {
      const rank = tables.bpe_ranks[pair];
      if (rank !== void 0 && rank < minRank) {
        minRank = rank;
        bigram = pair;
      }
    }
    if (bigram === null) {
      break;
    }
    const [first, second] = bigram.split(" ");
    const nextWord = [];
    let index = 0;
    while (index < word.length) {
      const found = word.indexOf(first, index);
      if (found < 0) {
        nextWord.push(...word.slice(index));
        break;
      }
      nextWord.push(...word.slice(index, found));
      index = found;
      if (word[index] === first && index < word.length - 1 && word[index + 1] === second) {
        nextWord.push(`${first}${second}`);
        index += 2;
      } else {
        nextWord.push((_a = word[index]) != null ? _a : "");
        index += 1;
      }
    }
    word = nextWord;
    if (word.length === 1) {
      break;
    }
    pairs = getPairs(word);
  }
  const result = word.join(" ");
  cache.set(token, result);
  return result;
}
var ClipBpeTokenizer = class {
  constructor(tables) {
    this.cache = /* @__PURE__ */ new Map();
    this.utf8 = new TextEncoder();
    this.tables = tables;
    this.cache.set("<start_of_text>", "<start_of_text>");
    this.cache.set("<end_of_text>", "<end_of_text>");
  }
  get contextLength() {
    return this.tables.context_length;
  }
  encode(text) {
    var _a;
    const cleaned = whitespaceClean(unescapeHtml(unescapeHtml(text))).toLowerCase();
    const tokens = [];
    for (const match of cleaned.matchAll(TOKEN_PATTERN)) {
      const mapped = Array.from(this.utf8.encode((_a = match[0]) != null ? _a : ""), (byte) => {
        var _a2;
        return (_a2 = this.tables.byte_encoder[String(byte)]) != null ? _a2 : "";
      }).join("");
      for (const bpeToken of applyBpe(mapped, this.tables, this.cache).split(" ")) {
        const id = this.tables.encoder[bpeToken];
        if (id !== void 0) {
          tokens.push(id);
        }
      }
    }
    return tokens;
  }
  tokenize(text) {
    const contextLength = this.tables.context_length;
    const ids = new Int32Array(contextLength);
    const tokens = [this.tables.sot_token_id, ...this.encode(text), this.tables.eot_token_id];
    if (tokens.length > contextLength) {
      tokens.length = contextLength;
      tokens[contextLength - 1] = this.tables.eot_token_id;
    }
    ids.set(tokens);
    return ids;
  }
};
async function loadClipTokenizer(source) {
  if (typeof source !== "string") {
    return new ClipBpeTokenizer(source);
  }
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to load SAM3 tokenizer tables from ${source}`);
  }
  return new ClipBpeTokenizer(await response.json());
}

// src/handler/sam3/sam3-pointer.ts
function pointerToImagePoint(event, target, options = {}) {
  var _a, _b, _c, _d, _e;
  const rect = isDomRect(target) ? target : target.getBoundingClientRect();
  const imageWidth = (_b = (_a = options.imageWidth) != null ? _a : target instanceof HTMLCanvasElement ? target.width : rect.width) != null ? _b : rect.width;
  const imageHeight = (_d = (_c = options.imageHeight) != null ? _c : target instanceof HTMLCanvasElement ? target.height : rect.height) != null ? _d : rect.height;
  const fit = (_e = options.objectFit) != null ? _e : "fill";
  if (fit === "contain" || fit === "cover") {
    const scale = fit === "contain" ? Math.min(rect.width / Math.max(imageWidth, 1e-6), rect.height / Math.max(imageHeight, 1e-6)) : Math.max(rect.width / Math.max(imageWidth, 1e-6), rect.height / Math.max(imageHeight, 1e-6));
    const drawWidth = imageWidth * scale;
    const drawHeight = imageHeight * scale;
    const offsetX = (rect.width - drawWidth) / 2;
    const offsetY = (rect.height - drawHeight) / 2;
    return {
      x: clamp3((event.clientX - rect.left - offsetX) / Math.max(scale, 1e-6), 0, imageWidth),
      y: clamp3((event.clientY - rect.top - offsetY) / Math.max(scale, 1e-6), 0, imageHeight)
    };
  }
  if (options.origin === "offset" && event.offsetX != null && event.offsetY != null) {
    const element = target instanceof HTMLElement ? target : null;
    const cssWidth = (element == null ? void 0 : element.clientWidth) || rect.width || 1;
    const cssHeight = (element == null ? void 0 : element.clientHeight) || rect.height || 1;
    return {
      x: clamp3(event.offsetX / cssWidth * imageWidth, 0, imageWidth),
      y: clamp3(event.offsetY / cssHeight * imageHeight, 0, imageHeight)
    };
  }
  return {
    x: clamp3(rect.width ? (event.clientX - rect.left) / rect.width * imageWidth : 0, 0, imageWidth),
    y: clamp3(rect.height ? (event.clientY - rect.top) / rect.height * imageHeight : 0, 0, imageHeight)
  };
}
function mapImagePoint(point, from, to) {
  return {
    x: clamp3(from.width ? point.x * to.width / from.width : point.x, 0, to.width),
    y: clamp3(from.height ? point.y * to.height / from.height : point.y, 0, to.height)
  };
}
function mapImageBox(box, from, to) {
  const topLeft = mapImagePoint({ x: box.left, y: box.top }, from, to);
  const bottomRight = mapImagePoint({ x: box.right, y: box.bottom }, from, to);
  return {
    left: Math.min(topLeft.x, bottomRight.x),
    top: Math.min(topLeft.y, bottomRight.y),
    right: Math.max(topLeft.x, bottomRight.x),
    bottom: Math.max(topLeft.y, bottomRight.y)
  };
}
function isDomRect(value) {
  return typeof DOMRect !== "undefined" && value instanceof DOMRect;
}
function clamp3(value, min, max) {
  if (!Number.isFinite(max) || max <= min) {
    return Math.max(min, value);
  }
  return Math.min(Math.max(value, min), max);
}

// src/handler/sam3/sam3-hover.ts
var Sam3HoverPreview = class {
  constructor(host, options = {}) {
    this.host = host;
    this.options = options;
    this.queued = null;
    this.queuedKind = "encoded";
    this.queuedDisplay = null;
    this.running = false;
    this.lastPoint = null;
    this.lastResult = null;
    var _a;
    this.onResult = (_a = options.onResult) != null ? _a : null;
  }
  get result() {
    return this.lastResult;
  }
  get mask() {
    var _a, _b;
    return (_b = (_a = this.lastResult) == null ? void 0 : _a.mask) != null ? _b : null;
  }
  get busy() {
    return this.running;
  }
  async idle() {
    while (this.running || this.queued) {
      await new Promise((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    }
  }
  /** 已编码图像坐标系中的点 */
  queuePoint(point) {
    this.queued = point;
    this.queuedKind = "encoded";
    this.queuedDisplay = null;
    this.kick();
  }
  /** 显示坐标系中的点（例如标注画布上的图像像素） */
  queueDisplayPoint(point, displayWidth, displayHeight) {
    this.queued = point;
    this.queuedKind = "display";
    this.queuedDisplay = { width: displayWidth, height: displayHeight };
    this.kick();
  }
  /** Pointer / Mouse 事件，自动映射到图像像素再编码 */
  queuePointer(event, target, pointer) {
    const size = pointerImageSize(target, pointer);
    const local = pointerToImagePoint(event, target, {
      ...pointer,
      imageWidth: size.width,
      imageHeight: size.height
    });
    this.queueDisplayPoint(local, size.width, size.height);
  }
  /** offsetX / offsetY（元素 CSS 像素） */
  queueOffset(offsetX, offsetY, target, pointer) {
    this.queuePointer({ clientX: 0, clientY: 0, offsetX, offsetY }, target, {
      ...pointer,
      origin: "offset"
    });
  }
  clear() {
    var _a;
    this.queued = null;
    this.lastPoint = null;
    this.lastResult = null;
    (_a = this.onResult) == null ? void 0 : _a.call(this, null);
  }
  /** 编码图坐标是否足够接近最近一次悬停结果，可直接作为点击确认。 */
  canReusePoint(point, maxDistance) {
    var _a, _b, _c;
    const limit = (_a = maxDistance != null ? maxDistance : this.options.confirmMaxDistance) != null ? _a : 8;
    const previous = (_b = this.lastResult) == null ? void 0 : _b.promptPoint;
    return Boolean(
      ((_c = this.lastResult) == null ? void 0 : _c.mask) && previous && Math.hypot(point.x - previous.x, point.y - previous.y) <= limit
    );
  }
  /**
   * 点击确认：距离最近悬停点足够近时直接返回预览掩码，否则按同一套 hover 后处理再推理。
   */
  async confirmPoint(point, maxDistance) {
    var _a;
    await this.idle();
    if (this.canReusePoint(point, maxDistance) && this.lastResult) {
      return this.lastResult;
    }
    const result = await this.host.hoverPoint(point, this.options);
    this.lastPoint = point;
    this.lastResult = result;
    (_a = this.onResult) == null ? void 0 : _a.call(this, result);
    return result;
  }
  kick() {
    if (!this.running) {
      void this.flush();
    }
  }
  async flush() {
    var _a, _b;
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
          const minMove = (_a = this.options.minMove) != null ? _a : 2;
          if (Math.hypot(point.x - this.lastPoint.x, point.y - this.lastPoint.y) < minMove) {
            continue;
          }
        }
        const result = kind === "display" && point && display ? await this.host.hoverFromDisplayPoint(point, display.width, display.height, this.options) : await this.host.hoverPoint(point, this.options);
        this.lastPoint = point;
        this.lastResult = result;
        (_b = this.onResult) == null ? void 0 : _b.call(this, result);
      }
    } finally {
      this.running = false;
      if (this.queued) {
        void this.flush();
      }
    }
  }
};
function pointerImageSize(target, options) {
  var _a, _b;
  if ((options == null ? void 0 : options.imageWidth) && options.imageHeight) {
    return { width: options.imageWidth, height: options.imageHeight };
  }
  if (target instanceof HTMLCanvasElement) {
    return { width: target.width, height: target.height };
  }
  const rect = typeof DOMRect !== "undefined" && target instanceof DOMRect ? target : target.getBoundingClientRect();
  return {
    width: (_a = options == null ? void 0 : options.imageWidth) != null ? _a : rect.width,
    height: (_b = options == null ? void 0 : options.imageHeight) != null ? _b : rect.height
  };
}

// src/handler/sam3/sam3-contours.ts
function packedMaskSize(mask) {
  if (mask.pixelMaskWidth && mask.pixelMaskHeight) {
    return {
      width: Math.max(1, Math.round(mask.pixelMaskWidth)),
      height: Math.max(1, Math.round(mask.pixelMaskHeight))
    };
  }
  return {
    width: Math.max(1, Math.round(mask.boundingBox.right - mask.boundingBox.left)),
    height: Math.max(1, Math.round(mask.boundingBox.bottom - mask.boundingBox.top))
  };
}
var DEFAULT_MAX_POLYGON_POINTS = 96;
var MAX_MASK_POLYGONS = 16;
var MIN_MASK_POLYGON_AREA_RATIO = 0.01;
function maskToPolygons(mask, options) {
  var _a, _b;
  const imageWidth = options.imageWidth;
  const imageHeight = options.imageHeight;
  if (imageWidth <= 0 || imageHeight <= 0) {
    return [];
  }
  const contours = tracePackedMaskContours(mask);
  if (contours.length === 0) {
    return [];
  }
  const sourceWidth = options.sourceWidth || imageWidth || 1;
  const sourceHeight = options.sourceHeight || imageHeight || 1;
  const size = packedMaskSize(mask);
  const box = mask.boundingBox;
  const boxWidth = box.right - box.left;
  const boxHeight = box.bottom - box.top;
  const maxPoints = (_a = options.maxPoints) != null ? _a : DEFAULT_MAX_POLYGON_POINTS;
  const mapped = contours.map((contour) => {
    const points = contour.map((point) => [
      clamp4((box.left + point[0] * boxWidth / size.width) * (imageWidth / sourceWidth), 0, imageWidth),
      clamp4((box.top + point[1] * boxHeight / size.height) * (imageHeight / sourceHeight), 0, imageHeight)
    ]);
    return { points, area: Math.abs(signedContourArea(points)) };
  }).filter((item) => item.points.length >= 3 && item.area > 0).sort((left, right) => right.area - left.area);
  if (mapped.length === 0) {
    return [];
  }
  if (options.prompt) {
    const mappedPrompt = {
      x: options.prompt.x * imageWidth / sourceWidth,
      y: options.prompt.y * imageHeight / sourceHeight
    };
    const selected = (_b = mapped.find((item) => pointInNumberContour(mappedPrompt, item.points))) != null ? _b : mapped[0];
    return [finalizePolygon(selected.points, imageWidth, imageHeight, maxPoints, options.epsilon)];
  }
  const minArea = mapped[0].area * MIN_MASK_POLYGON_AREA_RATIO;
  const outerContours = [];
  for (const candidate of mapped) {
    if (candidate.area < minArea || outerContours.length >= MAX_MASK_POLYGONS) {
      break;
    }
    const probe = { x: candidate.points[0][0], y: candidate.points[0][1] };
    if (outerContours.some((outer) => pointInNumberContour(probe, outer))) {
      continue;
    }
    outerContours.push(finalizePolygon(candidate.points, imageWidth, imageHeight, maxPoints, options.epsilon));
  }
  return outerContours.filter((polygon) => polygon.length >= 3);
}
function maskToPolygon(mask, options) {
  var _a;
  return (_a = maskToPolygons(mask, options)[0]) != null ? _a : [];
}
function tracePackedMaskContours(mask) {
  const packed = mask.bitPackedPixelMask;
  const { width, height } = packedMaskSize(mask);
  if (!(packed == null ? void 0 : packed.byteLength) || width * height > packed.byteLength * 8) {
    return [];
  }
  const vertexWidth = width + 1;
  const edges = [];
  const outgoing = /* @__PURE__ */ new Map();
  const isSet = (x, y) => {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return false;
    }
    const index = y * width + x;
    return (packed[index >> 3] & 1 << (index & 7)) !== 0;
  };
  const addEdge = (fromX, fromY, toX, toY, direction) => {
    const edge = { fromX, fromY, toX, toY, direction, used: false };
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
  const contours = [];
  for (const first of edges) {
    if (first.used) {
      continue;
    }
    const startKey = first.fromY * vertexWidth + first.fromX;
    const points = [[first.fromX, first.fromY]];
    let current = first;
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
function chooseNextBoundaryEdge(candidates, incomingDirection) {
  let selected;
  let selectedRank = Number.POSITIVE_INFINITY;
  for (const candidate of candidates != null ? candidates : []) {
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
function finalizePolygon(points, imageWidth, imageHeight, maxPoints, epsilon) {
  const pointLimit = Math.min(512, Math.max(3, Math.round(maxPoints)));
  const dense = simplifyClosedPolygon(points, 0);
  if (dense.length <= pointLimit) {
    return dense.length >= 3 ? dense : samplePolygon(points, pointLimit);
  }
  let lowerEpsilon = 0;
  let upperEpsilon = epsilon != null ? epsilon : polygonEpsilon(imageWidth, imageHeight);
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
function polygonEpsilon(width, height) {
  return Math.max(1.25, Math.max(width, height) / 2048);
}
function signedContourArea(points) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current[0] * next[1] - next[0] * current[1];
  }
  return area / 2;
}
function pointInNumberContour(point, contour) {
  let inside = false;
  for (let index = 0, previous = contour.length - 1; index < contour.length; previous = index, index += 1) {
    const current = contour[index];
    const last = contour[previous];
    if (current[1] > point.y !== last[1] > point.y && point.x < (last[0] - current[0]) * (point.y - current[1]) / (last[1] - current[1] || 1e-6) + current[0]) {
      inside = !inside;
    }
  }
  return inside;
}
function samplePolygon(points, limit) {
  if (points.length <= limit) {
    return points;
  }
  const sampled = [];
  const step = points.length / limit;
  for (let index = 0; index < limit; index += 1) {
    sampled.push(points[Math.floor(index * step)]);
  }
  return sampled;
}
function simplifyClosedPolygon(points, epsilon) {
  if (points.length <= 4) {
    return points;
  }
  const closed = [...points, points[0]];
  const simplified = simplifyRdp(closed, epsilon);
  if (simplified.length > 1 && simplified[0][0] === simplified[simplified.length - 1][0] && simplified[0][1] === simplified[simplified.length - 1][1]) {
    simplified.pop();
  }
  return simplified;
}
function simplifyRdp(points, epsilon) {
  if (points.length <= 4) {
    return points;
  }
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [startIndex, endIndex] = stack.pop();
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
function perpendicularDistance(point, start, end) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const length = Math.hypot(dx, dy);
  if (length <= 1e-6) {
    return Math.hypot(point[0] - start[0], point[1] - start[1]);
  }
  return Math.abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) / length;
}
function clamp4(value, min, max) {
  if (!Number.isFinite(max) || max <= min) {
    return Math.max(min, value);
  }
  return Math.min(Math.max(value, min), max);
}

// src/handler/sam3/sam3-mask.ts
function packedMaskSize2(mask) {
  if (mask.pixelMaskWidth && mask.pixelMaskHeight) {
    return {
      width: Math.max(1, Math.round(mask.pixelMaskWidth)),
      height: Math.max(1, Math.round(mask.pixelMaskHeight))
    };
  }
  const width = Math.max(1, Math.round(mask.boundingBox.right - mask.boundingBox.left));
  const height = Math.max(1, Math.round(mask.boundingBox.bottom - mask.boundingBox.top));
  return { width, height };
}
function pickBestMask(masks, ious = [], options = {}) {
  var _a, _b, _c, _d, _e, _f;
  if (masks.length === 0) {
    return null;
  }
  const pick = (_a = options.pick) != null ? _a : options.point ? "smallest" : "iou";
  if (pick === "first") {
    return (_b = masks[0]) != null ? _b : null;
  }
  if (pick === "smallest") {
    const containing = options.point ? masks.filter((mask) => pointInBox(options.point, mask.boundingBox)) : [];
    const pool = containing.length > 0 ? containing : [...masks];
    pool.sort((left, right) => maskArea(left) - maskArea(right));
    return (_c = pool[0]) != null ? _c : null;
  }
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < Math.max(ious.length, masks.length); index += 1) {
    const value = (_d = ious[index]) != null ? _d : Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return (_f = (_e = masks[bestIndex]) != null ? _e : masks[0]) != null ? _f : null;
}
function selectVisualMask(result, options = {}, promptBox) {
  var _a;
  const promptPoint = options.promptPoint;
  const picked = pickBestMask(result.masks, result.ious, {
    pick: (_a = options.pick) != null ? _a : promptPoint ? "smallest" : "iou",
    point: promptPoint
  });
  const maskIndex = picked ? Math.max(0, result.masks.indexOf(picked)) : 0;
  let mask = picked;
  const isolateAt = promptPoint != null ? promptPoint : promptBox ? {
    x: (promptBox.left + promptBox.right) / 2,
    y: (promptBox.top + promptBox.bottom) / 2
  } : void 0;
  if (mask && options.isolateComponent !== false && isolateAt) {
    mask = isolateMaskComponent(mask, isolateAt);
  }
  return {
    ...result,
    mask,
    maskIndex,
    promptPoint,
    promptBox: promptBox != null ? promptBox : void 0
  };
}
function isolateMaskComponent(mask, point) {
  const size = packedMaskSize2(mask);
  const width = size.width;
  const height = size.height;
  const box = mask.boundingBox;
  const boxWidth = Math.max(box.right - box.left, 1e-6);
  const boxHeight = Math.max(box.bottom - box.top, 1e-6);
  let startX = Math.round((point.x - box.left) / boxWidth * (width - 1));
  let startY = Math.round((point.y - box.top) / boxHeight * (height - 1));
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
  const cropped = new Uint8Array(Math.ceil(cropWidth * cropHeight / 8));
  for (let y = component.minY; y <= component.maxY; y += 1) {
    for (let x = component.minX; x <= component.maxX; x += 1) {
      if (!component.visited[y * width + x]) {
        continue;
      }
      const local = (y - component.minY) * cropWidth + (x - component.minX);
      cropped[local >> 3] |= 1 << (local & 7);
    }
  }
  const isolated = new Segmentation({
    label: mask.label,
    confidence: mask.confidence,
    boundingBox: {
      left: box.left + component.minX / width * boxWidth,
      top: box.top + component.minY / height * boxHeight,
      right: box.left + (component.maxX + 1) / width * boxWidth,
      bottom: box.top + (component.maxY + 1) / height * boxHeight
    },
    bitPackedPixelMask: cropped,
    pixelMaskWidth: cropWidth,
    pixelMaskHeight: cropHeight
  });
  isolated.segmentationEdgePoints = DrawTool.extractSegmentationEdgePoints(isolated);
  return isolated;
}
function maskToImagePixels(mask, options) {
  const imageWidth = options.imageWidth;
  const imageHeight = options.imageHeight;
  const sourceWidth = options.sourceWidth || imageWidth;
  const sourceHeight = options.sourceHeight || imageHeight;
  const size = packedMaskSize2(mask);
  const packed = mask.bitPackedPixelMask;
  if (imageWidth <= 0 || imageHeight <= 0 || packed.byteLength * 8 < size.width * size.height) {
    return null;
  }
  const displayBox = {
    left: mask.boundingBox.left * imageWidth / sourceWidth,
    top: mask.boundingBox.top * imageHeight / sourceHeight,
    right: mask.boundingBox.right * imageWidth / sourceWidth,
    bottom: mask.boundingBox.bottom * imageHeight / sourceHeight
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
          (top + y + 0.5 - displayBox.top) / Math.max(displayBox.bottom - displayBox.top, 1e-6) * size.height
        )
      )
    );
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(
        size.width - 1,
        Math.max(
          0,
          Math.floor(
            (left + x + 0.5 - displayBox.left) / Math.max(displayBox.right - displayBox.left, 1e-6) * size.width
          )
        )
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
function cropBinaryPixels(originX, originY, width, height, pixels, area) {
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
  const packed = new Uint8Array(Math.ceil(cropWidth * cropHeight / 8));
  let cropArea = 0;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      if (!pixels[y * width + x]) {
        continue;
      }
      const local = (y - minY) * cropWidth + (x - minX);
      cropped[local] = 1;
      packed[local >> 3] |= 1 << (local & 7);
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
    packed
  };
}
function floodFillMask(packed, width, height, startX, startY) {
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
    const y = index / width | 0;
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
function findNearestSetPixel(packed, width, height, startX, startY, maxRadius) {
  let best = null;
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
function pointInBox(point, box) {
  return point.x >= box.left && point.x <= box.right && point.y >= box.top && point.y <= box.bottom;
}
function maskArea(mask) {
  return Math.max(0, mask.boundingBox.right - mask.boundingBox.left) * Math.max(0, mask.boundingBox.bottom - mask.boundingBox.top);
}
function isPackedMaskSet(packed, pixelIndex) {
  if (pixelIndex < 0) {
    return false;
  }
  const byteIndex = pixelIndex >> 3;
  if (byteIndex >= packed.byteLength) {
    return false;
  }
  return (packed[byteIndex] & 1 << (pixelIndex & 7)) !== 0;
}

// src/sam3.ts
var DEFAULT_EXECUTION_PROVIDERS2 = ["wasm"];
function usesWebGpu(options) {
  var _a, _b, _c;
  const providers = (_c = (_b = (_a = options.sessionOptions) == null ? void 0 : _a.executionProviders) != null ? _b : options.executionProviders) != null ? _c : DEFAULT_EXECUTION_PROVIDERS2;
  return providers.some((provider) => {
    if (typeof provider === "string") {
      return provider === "webgpu";
    }
    return provider.name === "webgpu";
  });
}
var Sam3 = class _Sam3 {
  constructor(options) {
    this.options = options;
    this.visionSession = null;
    this.textSession = null;
    this.groundingSession = null;
    this.promptSession = null;
    this.handler = null;
    this.tokenizer = null;
    this.state = null;
    this.webGpu = usesWebGpu(options);
  }
  static async create(options) {
    const sam3 = new _Sam3(options);
    await sam3.load();
    return sam3;
  }
  get isLoaded() {
    return this.handler !== null;
  }
  get inferenceState() {
    return this.state;
  }
  async load() {
    var _a, _b, _c, _d, _e, _f, _g;
    const defaultNumThreads = this.webGpu ? 0 : 1;
    await ensureOnnxRuntimeWebInitialized({
      ...this.options,
      numThreads: (_a = this.options.numThreads) != null ? _a : defaultNumThreads
    });
    if (ort.env.wasm) {
      ort.env.wasm.numThreads = (_b = this.options.numThreads) != null ? _b : defaultNumThreads;
      ort.env.wasm.proxy = (_c = this.options.proxy) != null ? _c : false;
      ort.env.wasm.simd = true;
    }
    if (this.webGpu) {
      const webgpu = ort.env.webgpu;
      if (webgpu && webgpu.powerPreference === void 0) {
        webgpu.powerPreference = "high-performance";
      }
    }
    await this.dispose();
    try {
      const progress = this.options.onLoadProgress;
      progress == null ? void 0 : progress("\u6B63\u5728\u52A0\u8F7D\u89C6\u89C9\u7F16\u7801\u5668...");
      this.visionSession = await this.createSession("vision", this.options.visionEncoder);
      const promptSource = this.options.promptDecoder;
      const tokenizerPromise = this.options.tokenizer ? loadClipTokenizer(this.options.tokenizer) : Promise.resolve(null);
      if (this.webGpu) {
        progress == null ? void 0 : progress("\u6B63\u5728\u52A0\u8F7D\u6587\u672C\u7F16\u7801\u5668...");
        this.textSession = await this.createSession("text", this.options.textEncoder);
        progress == null ? void 0 : progress("\u6B63\u5728\u52A0\u8F7D Grounding \u89E3\u7801\u5668...");
        this.groundingSession = await this.createSession("grounding", this.options.groundingDecoder);
        if (promptSource) {
          progress == null ? void 0 : progress("\u6B63\u5728\u52A0\u8F7D Prompt \u89E3\u7801\u5668...");
          this.promptSession = await this.createSession("prompt", promptSource);
        }
        this.tokenizer = await tokenizerPromise;
      } else {
        progress == null ? void 0 : progress("\u6B63\u5728\u52A0\u8F7D\u6587\u672C / Grounding / Prompt / tokenizer...");
        const textPromise = this.createSession("text", this.options.textEncoder);
        const groundingPromise = this.createSession("grounding", this.options.groundingDecoder);
        const promptPromise = promptSource ? this.createSession("prompt", promptSource) : Promise.resolve(null);
        try {
          const [textSession, groundingSession, promptSession, tokenizer] = await Promise.all([
            textPromise,
            groundingPromise,
            promptPromise,
            tokenizerPromise
          ]);
          this.textSession = textSession;
          this.groundingSession = groundingSession;
          this.promptSession = promptSession;
          this.tokenizer = tokenizer;
        } catch (error) {
          const settled = await Promise.allSettled([textPromise, groundingPromise, promptPromise]);
          await Promise.all(
            settled.map(
              (item) => item.status === "fulfilled" && item.value ? item.value.release() : Promise.resolve()
            )
          );
          throw error;
        }
      }
      if (!this.tokenizer) {
        throw new Error("SAM3 tokenizer tables are required. Pass tokenizer: clip_bpe.json or parsed tables.");
      }
      this.handler = new Sam3Handler(
        this.visionSession,
        this.textSession,
        this.groundingSession,
        this.promptSession,
        this.tokenizer,
        {
          confidenceThreshold: (_d = this.options.confidenceThreshold) != null ? _d : 0.5,
          pixelConfidence: (_e = this.options.pixelConfidence) != null ? _e : 0.5,
          imageSize: (_f = this.options.imageSize) != null ? _f : SAM3_IMAGE_SIZE,
          maskSize: (_g = this.options.maskSize) != null ? _g : SAM3_MASK_SIZE,
          webGpu: this.webGpu
        }
      );
      return this;
    } catch (error) {
      await this.dispose();
      throw error;
    }
  }
  async setImage(image) {
    this.state = await this.ensureHandler().setImage(image, this.state);
    return this.state;
  }
  async setTextPrompt(prompt, description) {
    return this.ensureHandler().setTextPrompt(prompt, this.ensureState(), description);
  }
  async addGeometricPrompt(box, label = true) {
    return this.ensureHandler().addGeometricPrompt(box, label, this.ensureState());
  }
  async addGeometricPoint(point, label = true) {
    return this.ensureHandler().addGeometricPoint(point, label, this.ensureState());
  }
  async removeGeometricPrompt(index) {
    return this.ensureHandler().removeGeometricPrompt(index, this.ensureState());
  }
  async removeGeometricPoint(index) {
    return this.ensureHandler().removeGeometricPoint(index, this.ensureState());
  }
  resetPrompts() {
    return this.ensureHandler().resetPrompts(this.ensureState());
  }
  resetVisualPrompts() {
    return this.ensureHandler().resetVisualPrompts(this.ensureState());
  }
  async setConfidenceThreshold(threshold) {
    var _a;
    this.ensureHandler().setConfidenceThreshold(threshold);
    const state = this.ensureState();
    if (((_a = state.lastPcsRaw) == null ? void 0 : _a.length) || state.textEmbeddings) {
      return this.ensureHandler().applyConfidenceThreshold(state);
    }
    return state;
  }
  async predictConcept(prompt) {
    return this.ensureHandler().predictConcept(prompt, this.ensureState());
  }
  async predictVisual(prompt = {}) {
    return this.ensureHandler().predictVisual(prompt, this.ensureState());
  }
  async addPoint(point, label = 1) {
    return this.ensureHandler().addPoint(point, label, this.ensureState());
  }
  async addBox(box) {
    return this.ensureHandler().addBox(box, this.ensureState());
  }
  async addMask(mask) {
    return this.ensureHandler().addMask(mask, this.ensureState());
  }
  /**
   * 悬停/点选预览：默认每次独立单点，不写入 PVS 累积状态。
   * 填充请用 {@link drawMask}（像素掩码），不要把拼接边点当一个多边形描边。
   */
  async hoverPoint(point, options = {}) {
    var _a, _b;
    const label = (_a = options.label) != null ? _a : 1;
    if (options.mode === "accumulate") {
      const result = await this.addPoint(point, label);
      return this.selectHoverMask(result, {
        ...options,
        promptPoint: point
      });
    }
    return this.hoverVisual(
      {
        points: [{ point, label }],
        box: null,
        maskInput: options.refinePrevious ? void 0 : null,
        multimaskOutput: (_b = options.multimaskOutput) != null ? _b : true,
        persist: false
      },
      { ...options, promptPoint: point }
    );
  }
  async hoverBox(box, options = {}) {
    var _a, _b;
    return this.hoverVisual(
      {
        points: [],
        box,
        maskInput: options.refinePrevious ? void 0 : null,
        multimaskOutput: (_a = options.multimaskOutput) != null ? _a : false,
        persist: false
      },
      {
        ...options,
        promptPoint: (_b = options.promptPoint) != null ? _b : {
          x: (box.left + box.right) / 2,
          y: (box.top + box.bottom) / 2
        }
      },
      box
    );
  }
  async hoverPoints(points, options = {}) {
    var _a, _b, _c, _d, _e;
    const promptPoint = (_d = (_b = options.promptPoint) != null ? _b : (_a = points.find((item) => item.label === 1)) == null ? void 0 : _a.point) != null ? _d : (_c = points[0]) == null ? void 0 : _c.point;
    return this.hoverVisual(
      {
        points,
        box: null,
        maskInput: options.refinePrevious ? void 0 : null,
        multimaskOutput: (_e = options.multimaskOutput) != null ? _e : points.length <= 1,
        persist: false
      },
      { ...options, promptPoint }
    );
  }
  async hoverVisual(prompt, options = {}, promptBox) {
    var _a;
    const result = await this.predictVisual({
      ...prompt,
      persist: (_a = prompt.persist) != null ? _a : false
    });
    return this.selectHoverMask(result, options, promptBox != null ? promptBox : prompt.box);
  }
  /** PointerEvent / MouseEvent → 编码图坐标后悬停 */
  async hoverFromPointer(event, target, options = {}) {
    const size = this.pointerImageSize(target, options);
    const local = pointerToImagePoint(event, target, {
      ...options,
      imageWidth: size.width,
      imageHeight: size.height
    });
    return this.hoverFromDisplayPoint(local, size.width, size.height, options);
  }
  /** 显示画布坐标 → 编码图坐标后悬停 */
  async hoverFromDisplayPoint(point, displayWidth, displayHeight, options = {}) {
    return this.hoverPoint(this.toEncodedPoint(point, displayWidth, displayHeight), options);
  }
  /** offsetX / offsetY（CSS 像素）→ 编码图坐标后悬停 */
  async hoverFromOffset(offsetX, offsetY, target, options = {}) {
    return this.hoverFromPointer({ clientX: 0, clientY: 0, offsetX, offsetY }, target, {
      ...options,
      origin: "offset"
    });
  }
  createHoverPreview(options = {}) {
    return new Sam3HoverPreview(this, options);
  }
  toEncodedPoint(point, displayWidth, displayHeight) {
    const encoded = this.encodedSize();
    return mapImagePoint(point, { width: displayWidth, height: displayHeight }, encoded);
  }
  toEncodedBox(box, displayWidth, displayHeight) {
    const encoded = this.encodedSize();
    return mapImageBox(box, { width: displayWidth, height: displayHeight }, encoded);
  }
  toDisplayPoint(point, displayWidth, displayHeight) {
    const encoded = this.encodedSize();
    return mapImagePoint(point, encoded, { width: displayWidth, height: displayHeight });
  }
  static pointerToImagePoint(event, target, options = {}) {
    return pointerToImagePoint(event, target, options);
  }
  static mapPoint(point, from, to) {
    return mapImagePoint(point, from, to);
  }
  /**
   * 在已有 canvas 上下文上绘制像素掩码。边界用掩码边缘像素着色，
   * 不要把 `segmentationEdgePoints` 当单个闭合折线 fill/stroke。
   */
  drawMask(context, mask, options = {}) {
    var _a, _b, _c, _d, _e, _f, _g;
    const encoded = this.state ? { width: this.state.sourceWidth, height: this.state.sourceHeight } : {
      width: (_a = options.sourceWidth) != null ? _a : mask.boundingBox.right,
      height: (_b = options.sourceHeight) != null ? _b : mask.boundingBox.bottom
    };
    const sourceWidth = (_c = options.sourceWidth) != null ? _c : encoded.width;
    const sourceHeight = (_d = options.sourceHeight) != null ? _d : encoded.height;
    const displayWidth = (_e = options.displayWidth) != null ? _e : sourceWidth;
    const displayHeight = (_f = options.displayHeight) != null ? _f : sourceHeight;
    const box = mask.boundingBox;
    const dest = (_g = options.dest) != null ? _g : {
      left: box.left * displayWidth / Math.max(sourceWidth, 1e-6),
      top: box.top * displayHeight / Math.max(sourceHeight, 1e-6),
      right: box.right * displayWidth / Math.max(sourceWidth, 1e-6),
      bottom: box.bottom * displayHeight / Math.max(sourceHeight, 1e-6)
    };
    DrawTool.drawPackedMaskOverlay(context, mask, {
      fill: options.fill,
      stroke: options.stroke,
      dest
    });
  }
  maskToPolygon(mask, options) {
    return maskToPolygon(mask, this.withSourceSize(options));
  }
  maskToPolygons(mask, options) {
    return maskToPolygons(mask, this.withSourceSize(options));
  }
  toImagePixelMask(mask, imageWidth, imageHeight, options = {}) {
    var _a, _b, _c, _d;
    return maskToImagePixels(mask, {
      imageWidth,
      imageHeight,
      sourceWidth: (_b = options.sourceWidth) != null ? _b : (_a = this.state) == null ? void 0 : _a.sourceWidth,
      sourceHeight: (_d = options.sourceHeight) != null ? _d : (_c = this.state) == null ? void 0 : _c.sourceHeight
    });
  }
  selectVisualMask(result, options = {}, promptBox) {
    return selectVisualMask(result, options, promptBox);
  }
  /** 把 hover / confirm 选中的候选写入 PVS 状态，不再次推理。 */
  acceptVisualResult(result) {
    var _a, _b;
    const state = this.ensureState();
    if (result.promptBox) {
      state.pvsBox = result.promptBox;
      state.pvsPoints = [];
    } else if (result.promptPoint) {
      state.pvsPoints = [{ point: result.promptPoint, label: 1 }];
      state.pvsBox = null;
    }
    state.lastPvs = {
      masks: result.masks,
      lowResMasks: result.lowResMasks,
      ious: result.ious,
      objectScores: result.objectScores,
      maskWidth: result.maskWidth,
      maskHeight: result.maskHeight
    };
    const index = result.maskIndex >= 0 ? result.maskIndex : 0;
    const logits = result.lowResMasks[index];
    if (logits) {
      state.pvsMaskInput = {
        logits,
        width: (_a = result.maskWidth) != null ? _a : SAM3_MASK_SIZE,
        height: (_b = result.maskHeight) != null ? _b : SAM3_MASK_SIZE
      };
    }
    return result;
  }
  selectVisualCandidate(index) {
    return this.ensureHandler().selectPvsCandidate(index, this.ensureState());
  }
  async removePoint(index) {
    return this.ensureHandler().removePoint(index, this.ensureState());
  }
  drawSegmentations(source, segmentations, canvas, options = {}) {
    DrawTool.drawSegmentationEdgePoints(source, segmentations, canvas, {
      drawBoundingBoxes: true,
      drawLabel: true,
      drawSegmentationPixelMask: true,
      fillSegmentationEdgePoints: true,
      ...options
    });
  }
  drawSegmentationEdgePoints(source, segmentations, canvas, options = {}) {
    DrawTool.drawSegmentationEdgePoints(source, segmentations, canvas, options);
  }
  async dispose() {
    var _a;
    (_a = this.handler) == null ? void 0 : _a.dispose();
    const sessions = [this.visionSession, this.textSession, this.groundingSession, this.promptSession];
    this.visionSession = null;
    this.textSession = null;
    this.groundingSession = null;
    this.promptSession = null;
    this.handler = null;
    this.tokenizer = null;
    this.state = null;
    await Promise.all(sessions.filter(Boolean).map((session) => session == null ? void 0 : session.release()));
  }
  createSessionOptions(kind) {
    var _a, _b, _c, _d, _e;
    const userOptions = (_a = this.options.sessionOptions) != null ? _a : {};
    const userExtra = (_b = userOptions.extra) != null ? _b : {};
    const userSession = (_c = userExtra.session) != null ? _c : {};
    const keepVisionOnGpu = this.webGpu && kind === "vision";
    const disableGraphOpt = kind === "vision" || kind === "text";
    const options = {
      enableCpuMemArena: true,
      enableMemPattern: true,
      ...userOptions,
      graphOptimizationLevel: (_d = userOptions.graphOptimizationLevel) != null ? _d : disableGraphOpt ? "disabled" : "all",
      extra: {
        ...userExtra,
        session: {
          strict_shape_type_inference: "0",
          ...keepVisionOnGpu ? { use_device_allocator_for_initializers: "1" } : {},
          ...userSession
        }
      },
      executionProviders: (_e = userOptions.executionProviders) != null ? _e : this.createExecutionProviders()
    };
    if (keepVisionOnGpu && options.preferredOutputLocation == null) {
      options.preferredOutputLocation = "gpu-buffer";
    }
    return options;
  }
  createExecutionProviders() {
    var _a;
    const providers = [...(_a = this.options.executionProviders) != null ? _a : DEFAULT_EXECUTION_PROVIDERS2];
    if (!this.webGpu) {
      return providers;
    }
    return providers.map((provider) => {
      if (provider === "webgpu") {
        return { name: "webgpu", preferredLayout: "NCHW", validationMode: "wgpuOnly" };
      }
      return provider;
    });
  }
  async createSession(kind, model) {
    const label = `${kind}-encoder`;
    const options = this.createSessionOptions(kind);
    try {
      return await this.createSessionWithOptions(model, options);
    } catch (error) {
      if (kind === "vision" && this.webGpu && options.preferredOutputLocation === "gpu-buffer") {
        try {
          return await this.createSessionWithOptions(model, { ...options, preferredOutputLocation: void 0 });
        } catch (e) {
        }
      }
      if (options.graphOptimizationLevel === "all") {
        try {
          return await this.createSessionWithOptions(model, { ...options, graphOptimizationLevel: "disabled" });
        } catch (e) {
        }
      }
      const message = error instanceof Error ? error.message : String(error);
      if (/bad_alloc/i.test(message)) {
        throw new Error(
          `Failed to create SAM3 session "${label}": \u6D4F\u89C8\u5668 WASM \u5185\u5B58\u4E0D\u8DB3\u3002\u8BF7\u52A0\u8F7D fp16 \u6A21\u578B\uFF08vision-encoder.fp16.onnx / text-encoder.fp16.onnx\uFF09\uFF0C\u5E76\u4F18\u5148\u4F7F\u7528 WebGPU\u3002\u539F\u59CB\u9519\u8BEF: ${message}`
        );
      }
      throw new Error(`Failed to create SAM3 session "${label}": ${message}`);
    }
  }
  async createSessionWithOptions(model, options) {
    return createOrtInferenceSession(model, options);
  }
  encodedSize() {
    const state = this.ensureState();
    return { width: state.sourceWidth, height: state.sourceHeight };
  }
  pointerImageSize(target, options) {
    var _a, _b;
    if (options.imageWidth && options.imageHeight) {
      return { width: options.imageWidth, height: options.imageHeight };
    }
    if (target instanceof HTMLCanvasElement) {
      return { width: target.width, height: target.height };
    }
    const rect = typeof DOMRect !== "undefined" && target instanceof DOMRect ? target : target.getBoundingClientRect();
    return { width: (_a = options.imageWidth) != null ? _a : rect.width, height: (_b = options.imageHeight) != null ? _b : rect.height };
  }
  selectHoverMask(result, options, promptBox) {
    return selectVisualMask(result, options, promptBox);
  }
  withSourceSize(options) {
    var _a, _b, _c, _d;
    return {
      ...options,
      sourceWidth: (_b = options.sourceWidth) != null ? _b : (_a = this.state) == null ? void 0 : _a.sourceWidth,
      sourceHeight: (_d = options.sourceHeight) != null ? _d : (_c = this.state) == null ? void 0 : _c.sourceHeight
    };
  }
  ensureHandler() {
    if (!this.handler) {
      throw new Error("SAM3 is not loaded. Call Sam3.create() / load() first.");
    }
    return this.handler;
  }
  ensureState() {
    if (!this.state) {
      throw new Error("You must call setImage() before prompting SAM3.");
    }
    return this.state;
  }
};

export { Classification, DrawTool, OBBDetection, ObjectDetection, PoseEstimation, SAM3_IMAGE_SIZE, SAM3_MASK_SIZE, SAM3_TEXT_LENGTH, Sam3, Sam3HoverPreview, Segmentation, TrackingInfo, Yolo, YoloExecutionProviderNames, YoloExecutionProviderOptions, YoloWebExecutionProviderOptions, canReuseOrtBundle, ensureOnnxRuntimeWebInitialized, getLoadedOrtBundle, getOrt, initializeOnnxRuntimeWeb, isWebAssemblyJspiAvailable, isolateMaskComponent, mapImageBox, mapImagePoint, maskToImagePixels, maskToPolygon, maskToPolygons, ort, pickBestMask, pointerToImagePoint, resolveOrtBundle, selectVisualMask, splitTextPrompts };
//# sourceMappingURL=index.js.map
//# sourceMappingURL=index.js.map