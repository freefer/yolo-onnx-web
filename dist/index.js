import * as ort from 'onnxruntime-web/all';
export { ort };

// src/runtime.ts
var configured = false;
function initializeOnnxRuntimeWeb(options = {}) {
  if (options.wasmPaths !== void 0) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.numThreads !== void 0) {
    ort.env.wasm.numThreads = options.numThreads;
  }
  if (options.proxy !== void 0) {
    ort.env.wasm.proxy = options.proxy;
  }
  configured = true;
}
function ensureOnnxRuntimeWebInitialized(options = {}) {
  if (configured) {
    initializeOnnxRuntimeWeb(options);
    return;
  }
  initializeOnnxRuntimeWeb(options);
}

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
    super(options);
    this.bitPackedPixelMask = options.bitPackedPixelMask;
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
    this._yolo = yolo;
  }
  preprocessImage(img, roi = null) {
    var _a, _b, _c;
    const inputShape = this.getInputShape();
    const [, channels, modelHeight, modelWidth] = inputShape;
    const sourceRect = this.getSourceRect(img, roi);
    const resizeMode = (_a = this._yolo.yoloOptions.imageResize) != null ? _a : "stretch";
    const { drawWidth, drawHeight, xPad, yPad, gain } = resizeMode === "stretch" ? { drawWidth: modelWidth, drawHeight: modelHeight, xPad: 0, yPad: 0, gain: 1 } : this.calculateProportionalResize(sourceRect.width, sourceRect.height, modelWidth, modelHeight);
    const context = this.getPreprocessContext(sourceRect.width, sourceRect.height);
    context.clearRect(0, 0, sourceRect.width, sourceRect.height);
    context.drawImage(
      img,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      0,
      0,
      sourceRect.width,
      sourceRect.height
    );
    const imageData = context.getImageData(0, 0, sourceRect.width, sourceRect.height).data;
    const pixelCount = modelWidth * modelHeight;
    const tensorData = this.getPreprocessTensorData(channels * pixelCount);
    const imageMean = (_b = this._yolo.yoloOptions.imageMean) != null ? _b : DEFAULT_IMAGE_MEAN;
    const imageStd = (_c = this._yolo.yoloOptions.imageStd) != null ? _c : DEFAULT_IMAGE_STD;
    this.writePreprocessedTensor(
      imageData,
      sourceRect.width,
      sourceRect.height,
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
  async RunObjectDetection(img, confidence, iou, roi = null) {
    var _a, _b;
    if (this._yolo.onnxModel.modelType !== "ObjectDetection") {
      unsupportedTask("ObjectDetection");
    }
    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor("float32", input.tensorData, input.inputShape)
    });
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
    unsupportedTask("Segmentation");
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
    const labels = backgroundClassIndex >= 0 ? this._yolo.onnxModel.labels.filter((label) => label.index !== backgroundClassIndex).map((label, index) => ({ ...label, index })) : this._yolo.onnxModel.labels;
    const maxDetections = predictions;
    const { topIndices, topScores } = this.getTopKBuffers(maxDetections);
    const objects = [];
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
      const mappedLabelIndex = backgroundClassIndex >= 0 && labelIndex > backgroundClassIndex ? labelIndex - 1 : labelIndex;
      const label = labels[mappedLabelIndex];
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
    ensureOnnxRuntimeWebInitialized(options);
  }
  get yoloOptions() {
    return this.options;
  }
  static async create(options) {
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
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    this.drawBoundingBoxes(context, detections, width, height, options);
  }
  drawClassifications(source, classifications, canvas, options = {}) {
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
  drawObbDetections(source, detections, canvas, options = {}) {
    var _a, _b, _c, _d, _e, _f;
    const { context, width, height } = this.prepareDrawingCanvas(source, canvas, options.drawSource);
    const font = (_a = options.font) != null ? _a : `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const lineWidth = (_b = options.lineWidth) != null ? _b : Math.max(2, Math.round(Math.min(width, height) / 320));
    const drawLabel = (_c = options.drawLabel) != null ? _c : true;
    const drawConfidenceScore = (_d = options.drawConfidenceScore) != null ? _d : true;
    const drawLabelBackground = (_e = options.drawLabelBackground) != null ? _e : true;
    const colors = (_f = options.boundingBoxHexColors) != null ? _f : [...DEFAULT_BOX_COLORS];
    context.font = font;
    context.textBaseline = "middle";
    context.lineWidth = lineWidth;
    for (const detection of detections) {
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, options.boundingBoxOpacity);
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
          fontColor: options.fontColor
        });
      }
    }
  }
  drawSegmentations(source, segmentations, canvas, options = {}) {
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
      for (const segmentation of segmentations) {
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
  drawPoseEstimations(source, poseEstimations, canvas, options = {}) {
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
  drawBoundingBoxes(context, detections, width, height, options = {}) {
    var _a, _b, _c, _d, _e, _f;
    const colors = (_a = options.boundingBoxHexColors) != null ? _a : [...DEFAULT_BOX_COLORS];
    const lineWidth = (_b = options.lineWidth) != null ? _b : Math.max(2, Math.round(Math.min(width, height) / 320));
    const font = (_c = options.font) != null ? _c : `${Math.max(14, Math.round(Math.min(width, height) / 45))}px Arial`;
    const drawLabel = (_d = options.drawLabel) != null ? _d : true;
    const drawConfidenceScore = (_e = options.drawConfidenceScore) != null ? _e : true;
    const drawLabelBackground = (_f = options.drawLabelBackground) != null ? _f : true;
    context.lineWidth = lineWidth;
    context.font = font;
    context.textBaseline = "middle";
    for (const detection of detections) {
      const { left, top, right, bottom } = detection.boundingBox;
      const boxWidth = right - left;
      const boxHeight = bottom - top;
      const color = this.getDetectionColor(detection, colors, options.strokeStyle, options.boundingBoxOpacity);
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
          fontColor: options.fontColor
        });
      }
    }
  }
  prepareDrawingCanvas(source, canvas, drawSource = true) {
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
      context.drawImage(source, 0, 0, width, height);
    }
    return { context, width, height };
  }
  getDetectionColor(detection, colors, fallback, alpha = 255) {
    var _a;
    const color = (_a = fallback != null ? fallback : colors[detection.label.index % colors.length]) != null ? _a : DEFAULT_BOX_COLORS[0];
    return this.withAlpha(color, alpha);
  }
  withAlpha(color, alpha) {
    if (!color.startsWith("#") || color.length !== 7) {
      return color;
    }
    const r = Number.parseInt(color.slice(1, 3), 16);
    const g = Number.parseInt(color.slice(3, 5), 16);
    const b = Number.parseInt(color.slice(5, 7), 16);
    const normalizedAlpha = this.clamp(alpha, 0, 255) / 255;
    return `rgba(${r}, ${g}, ${b}, ${normalizedAlpha})`;
  }
  drawDetectionLabel(context, detection, x, y, backgroundColor, options) {
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
  getCanvasFontSize(font) {
    const match = font.match(/(\d+(?:\.\d+)?)px/);
    return match ? Number(match[1]) : 14;
  }
  getObbCorners(box, radians) {
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
  drawSegmentationMask(context, segmentation, color) {
    const { left, top, right, bottom } = segmentation.boundingBox;
    const width = right - left;
    const height = bottom - top;
    if (width <= 0 || height <= 0 || segmentation.bitPackedPixelMask.byteLength === 0) {
      return;
    }
    const imageData = context.createImageData(width, height);
    const rgba = this.parseCanvasColor(color);
    const maskCanvas = document.createElement("canvas");
    const maskContext = maskCanvas.getContext("2d");
    if (!maskContext) {
      throw new Error("Canvas 2D context is not available.");
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
  drawSegmentationContour(context, segmentation, color, thickness) {
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
        const isEdge = x === 0 || x === width - 1 || y === 0 || y === height - 1 || !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex - 1) || !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex + 1) || !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex - width) || !this.isPackedMaskSet(segmentation.bitPackedPixelMask, pixelIndex + width);
        if (isEdge) {
          context.fillRect(left + x, top + y, thickness, thickness);
        }
      }
    }
  }
  isPackedMaskSet(mask, pixelIndex) {
    if (pixelIndex < 0) {
      return false;
    }
    const byteIndex = pixelIndex >> 3;
    if (byteIndex >= mask.byteLength) {
      return false;
    }
    return (mask[byteIndex] & 1 << (pixelIndex & 7)) !== 0;
  }
  parseCanvasColor(color) {
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
  drawPoseConnections(context, keyPoints, confidence, markers, defaultColor) {
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
  drawPoseConnection(context, source, target, confidence, color) {
    if (!source || !target || source.confidence < confidence || target.confidence < confidence) {
      return;
    }
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(source.x, source.y);
    context.lineTo(target.x, target.y);
    context.stroke();
  }
  tensor(type, data, dims) {
    return new ort.Tensor(type, data, dims);
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
    const options = this.createSessionOptions();
    if (typeof model === "string") {
      return ort.InferenceSession.create(model, options);
    }
    if (model instanceof Uint8Array) {
      return ort.InferenceSession.create(model, options);
    }
    return ort.InferenceSession.create(model, options);
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
      RFDETR: ["ObjectDetection"],
      WORLDV2: ["ObjectDetection"]
    };
    return supportMap[modelVersion].includes(modelType);
  }
};

export { Classification, OBBDetection, ObjectDetection, PoseEstimation, Segmentation, TrackingInfo, Yolo, YoloExecutionProviderNames, YoloExecutionProviderOptions, YoloWebExecutionProviderOptions, initializeOnnxRuntimeWeb };
//# sourceMappingURL=index.js.map
//# sourceMappingURL=index.js.map