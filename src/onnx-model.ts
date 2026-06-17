import type * as OrtTypes from 'onnxruntime-web';
import type { LabelModel, ModelDataType, ModelType, ModelVersion, OnnxModel, YoloModelSource } from './types';

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
): Promise<OnnxModel> {
  const customMetaData = await parseCustomMetadata(model);
  const inputShapes = getShapes(session.inputMetadata);
  const outputShapes = getShapes(session.outputMetadata);
  const firstInputShape = Object.values(inputShapes)[0] ?? [];

  return {
    inputShapes,
    outputShapes,
    customMetaData,
    modelDataType: getModelDataType(session.inputMetadata),
    modelType: getModelType(requireMetadata(customMetaData, 'task')),
    modelVersion: getModelVersion(requireMetadata(customMetaData, 'description')),
    labels: mapLabelsAndColors(requireMetadata(customMetaData, 'names')),
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

function getModelType(modelType: string): ModelType {
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

function getModelVersion(modelDescription: string): ModelVersion {
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

function requireMetadata(metadata: Record<string, string>, key: string): string {
  const value = metadata[key];

  if (value === undefined) {
    throw new Error(`ONNX custom metadata "${key}" is missing.`);
  }

  return value;
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
