import type * as OrtTypes from 'onnxruntime-web';
import type { LabelModel, ModelDataType, ModelType, ModelVersion, OnnxModel, YoloLabels, YoloModelSource, YoloOptions } from './types';

interface ProtobufField {
  fieldNumber: number;
  wireType: number;
  value?: Uint8Array;
}

interface ReaderState {
  offset: number;
}

const MODEL_METADATA_PROPS_FIELD = 14;
const STRING_ENTRY_KEY_FIELD = 1;
const STRING_ENTRY_VALUE_FIELD = 2;

export async function parseOnnxModel(
  session: OrtTypes.InferenceSession,
  model: YoloModelSource,
  options: Pick<YoloOptions, 'labels' | 'modelType' | 'modelVersion'> = {},
): Promise<OnnxModel> {
  const customMetaData = await parseCustomMetadata(model);
  const inputShapes = getShapes(session.inputMetadata);
  const outputShapes = getShapes(session.outputMetadata);
  const firstInputShape = Object.values(inputShapes)[0] ?? [];
  const labels = options.labels
    ? mapLabels(parseLabelsInput(options.labels))
    : customMetaData.names
      ? mapLabelsAndColors(customMetaData.names)
      : inferLabels(outputShapes);

  return {
    inputShapes,
    outputShapes,
    customMetaData,
    modelDataType: getModelDataType(session.inputMetadata),
    modelType: options.modelType ?? getModelType(customMetaData, outputShapes),
    modelVersion: options.modelVersion ?? getModelVersion(customMetaData, outputShapes),
    labels,
    inputShapeSize: calculateTotalInputShapeSize(firstInputShape),
  };
}

async function parseCustomMetadata(model: YoloModelSource): Promise<Record<string, string>> {
  const bytes = await getModelBytes(model);
  const metadata: Record<string, string> = {};

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

async function getModelBytes(model: YoloModelSource): Promise<Uint8Array> {
  if (typeof model === 'string') {
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

function getShapes(metadata: readonly OrtTypes.InferenceSession.ValueMetadata[]): Record<string, number[]> {
  const shapes: Record<string, number[]> = {};

  for (const item of metadata) {
    shapes[item.name] = item.isTensor ? item.shape.map(normalizeDimension) : [];
  }

  return shapes;
}

function normalizeDimension(dimension: number | string): number {
  return typeof dimension === 'number' ? dimension : -1;
}

function getModelDataType(metadata: readonly OrtTypes.InferenceSession.ValueMetadata[]): ModelDataType {
  const firstTensor = metadata.find(item => item.isTensor);

  return firstTensor?.isTensor && firstTensor.type === 'float16' ? 'Float16' : 'Float';
}

function getModelType(metadata: Record<string, string>, outputShapes: Record<string, number[]>): ModelType {
  if (metadata.task) {
    return getModelTypeFromMetadata(metadata.task);
  }

  if (isRfdetrSegmentationOutput(outputShapes)) {
    return 'Segmentation';
  }

  if (isRfdetrOutput(outputShapes)) {
    return 'ObjectDetection';
  }

  throw new Error('Unsupported task');
}

function getModelTypeFromMetadata(modelType: string): ModelType {
  switch (modelType) {
    case 'classify':
      return 'Classification';
    case 'detect':
      return 'ObjectDetection';
    case 'obb':
      return 'ObbDetection';
    case 'segment':
      return 'Segmentation';
    case 'pose':
      return 'PoseEstimation';
    default:
      throw new Error('Unsupported task');
  }
}

function getModelVersion(metadata: Record<string, string>, outputShapes: Record<string, number[]>): ModelVersion {
  if (metadata.description) {
    return getModelVersionFromDescription(metadata.description);
  }

  if (isRfdetrOutput(outputShapes)) {
    return 'RFDETR';
  }

  throw new Error('Onnx model not supported!');
}

function getModelVersionFromDescription(modelDescription: string): ModelVersion {
  const version = modelDescription.toLowerCase();

  if (version.startsWith('ultralytics yolov5')) return 'V5U';
  if (version.startsWith('ultralytics yolov8')) return 'V8';
  if (version.startsWith('ultralytics yoloe-v8')) return 'V8E';
  if (version.startsWith('ultralytics yolov9')) return 'V9';
  if (version.startsWith('ultralytics yolov10')) return 'V10';
  if (version.startsWith('ultralytics yolo11')) return 'V11';
  if (version.startsWith('ultralytics yoloe-11')) return 'V11E';
  if (version.startsWith('ultralytics yolov12')) return 'V12';
  if (version.startsWith('ultralytics yolo26')) return 'V26';
  if (version.includes('worldv2')) return 'WORLDV2';
  if (version.startsWith('ultralytics rt-detr')) return 'RTDETR';
  if (version.startsWith('ultralytics') && !version.includes('yolo')) return 'V8';

  throw new Error('Onnx model not supported!');
}

function mapLabelsAndColors(onnxLabelData: string): LabelModel[] {
  const labels = parseLabels(onnxLabelData);

  return Object.entries(labels).map(([, name], index) => ({
    index,
    name,
  }));
}

function mapLabels(labels: readonly string[]): LabelModel[] {
  return labels.map((name, index) => ({ index, name }));
}

function parseLabelsInput(labels: YoloLabels): string[] {
  if (typeof labels !== 'string') {
    return labels.map(label => label.trim()).filter(Boolean);
  }

  const content = labels.trim();

  if (!content) {
    return [];
  }

  try {
    const parsed = JSON.parse(content);

    if (Array.isArray(parsed)) {
      return parsed.map(String).map(label => label.trim()).filter(Boolean);
    }
  } catch {
    // Plain class_names.txt content is expected for most RF-DETR models.
  }

  return content
    .split(/\r?\n|,/)
    .map(label => label.trim())
    .filter(Boolean);
}

function inferLabels(outputShapes: Record<string, number[]>): LabelModel[] {
  const classCount = tryGetRfdetrClassCount(outputShapes);

  if (classCount === null) {
    throw new Error('ONNX custom metadata "names" is missing. Pass labels in YoloOptions for metadata-free models.');
  }

  return Array.from({ length: classCount }, (_, index) => ({
    index,
    name: index === 0 ? 'background_class' : `class_${index}`,
  }));
}

function parseLabels(onnxLabelData: string): Record<number, string> {
  const labels: Record<number, string> = {};
  const content = onnxLabelData.trim().replace(/^\{|\}$/g, '');

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

function calculateTotalInputShapeSize(shape: number[]): number {
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

function isRfdetrOutput(outputShapes: Record<string, number[]>): boolean {
  const dets = outputShapes.dets;
  const labels = outputShapes.labels;

  return Boolean(dets && labels && dets.length === 3 && labels.length === 3 && dets[2] === 4);
}

function isRfdetrSegmentationOutput(outputShapes: Record<string, number[]>): boolean {
  const masks = outputShapes.masks;

  return Boolean(isRfdetrOutput(outputShapes) && masks && masks.length === 4);
}

function tryGetRfdetrClassCount(outputShapes: Record<string, number[]>): number | null {
  return isRfdetrOutput(outputShapes) ? outputShapes.labels[2] : null;
}

function parseStringStringEntry(bytes: Uint8Array): { key: string; value: string } {
  let key = '';
  let value = '';

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

function readFields(bytes: Uint8Array): ProtobufField[] {
  const state: ReaderState = { offset: 0 };
  const fields: ProtobufField[] = [];

  while (state.offset < bytes.length) {
    const tag = readVarint(bytes, state);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 7;

    fields.push({
      fieldNumber,
      wireType,
      value: readFieldValue(bytes, state, wireType),
    });
  }

  return fields;
}

function readFieldValue(bytes: Uint8Array, state: ReaderState, wireType: number): Uint8Array | undefined {
  switch (wireType) {
    case 0:
      readVarint(bytes, state);
      return undefined;
    case 1:
      state.offset += 8;
      return undefined;
    case 2: {
      const length = readVarint(bytes, state);
      const start = state.offset;
      state.offset += length;
      return bytes.subarray(start, start + length);
    }
    case 5:
      state.offset += 4;
      return undefined;
    default:
      throw new Error(`Unsupported ONNX protobuf wire type: ${wireType}`);
  }
}

function readVarint(bytes: Uint8Array, state: ReaderState): number {
  let result = 0;
  let shift = 0;

  while (state.offset < bytes.length) {
    const byte = bytes[state.offset++];
    result += (byte & 0x7f) * 2 ** shift;

    if ((byte & 0x80) === 0) {
      return result;
    }

    shift += 7;
  }

  throw new Error('Invalid ONNX protobuf varint.');
}

function decodeUtf8(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}
