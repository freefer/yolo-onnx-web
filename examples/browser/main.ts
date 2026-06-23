import { Yolo, YoloWebExecutionProviderOptions } from '../../src';
import type {
  Classification,
  ModelType,
  OBBDetection,
  ObjectDetection,
  OnnxModel,
  PoseEstimation,
  Segmentation,
  YoloExecutionProvider,
  YoloImageSource,
  YoloModelSource,
} from '../../src';

const ORT_WASM_PATHS = '/examples/browser/ort-wasm/';
const DEFAULT_MODEL_URL = '/examples/model/yolo26s.onnx';
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 640;

const backendSelect = getElement<HTMLSelectElement>('#backend');
const inputModeSelect = getElement<HTMLSelectElement>('#inputMode');
const modelUrlInput = getElement<HTMLInputElement>('#modelUrl');
const modelFileInput = getElement<HTMLInputElement>('#modelFile');
const classNamesFileInput = getElement<HTMLInputElement>('#classNamesFile');
const confidenceInput = getElement<HTMLInputElement>('#confidenceInput');
const iouInput = getElement<HTMLInputElement>('#iouInput');
const confidenceValue = getElement<HTMLElement>('#confidenceValue');
const iouValue = getElement<HTMLElement>('#iouValue');
const imageFileInput = getElement<HTMLInputElement>('#imageFile');
const imageInputGroup = getElement<HTMLElement>('#imageInputGroup');
const startCameraButton = getElement<HTMLButtonElement>('#startCameraButton');
const stopCameraButton = getElement<HTMLButtonElement>('#stopCameraButton');
const cameraVideo = getElement<HTMLVideoElement>('#cameraVideo');
const runButton = getElement<HTMLButtonElement>('#runButton');
const viewer = getElement<HTMLElement>('#viewer');
const preview = getElement<HTMLCanvasElement>('#preview');
const fpsBadge = getElement<HTMLElement>('#fpsBadge');
const modelInfo = getElement<HTMLPreElement>('#modelInfo');
const output = getElement<HTMLPreElement>('#output');

let yolo: Yolo | null = null;
let activeExecutionProviders: readonly YoloExecutionProvider[] = [];
let loadingModelToken = 0;
let cameraStream: MediaStream | null = null;
let isCameraInferencing = false;
let cameraInferenceFrame = 0;
let fpsLastTimestamp = 0;
let fpsAverage = 0;
let thresholdPreviewTimer = 0;
let isThresholdPreviewRunning = false;
let pendingThresholdPreview = false;

type InferenceResult =
  | { modelType: 'Classification'; result: Classification[] }
  | { modelType: 'ObjectDetection'; result: ObjectDetection[] }
  | { modelType: 'ObbDetection'; result: OBBDetection[] }
  | { modelType: 'Segmentation'; result: Segmentation[] }
  | { modelType: 'PoseEstimation'; result: PoseEstimation[] };

renderBackendOptions();

inputModeSelect.addEventListener('change', () => {
  updateInputMode();
});

backendSelect.addEventListener('change', () => {
  if (!hasModelSource()) {
    return;
  }

  loadSelectedModel().catch(error => {
    writeModelInfo(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

modelFileInput.addEventListener('change', () => {
  if (modelFileInput.files?.[0]) {
    modelUrlInput.value = '';
  }

  loadSelectedModel().catch(error => {
    writeModelInfo(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

classNamesFileInput.addEventListener('change', () => {
  if (!hasModelSource()) {
    return;
  }

  loadSelectedModel().catch(error => {
    writeModelInfo(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

imageFileInput.addEventListener('change', () => {
  updateRunButton();
});

modelUrlInput.addEventListener('change', () => {
  if (modelUrlInput.value.trim()) {
    modelFileInput.value = '';
  }

  loadSelectedModel().catch(error => {
    writeModelInfo(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

modelUrlInput.addEventListener('keydown', event => {
  if (event.key !== 'Enter') {
    return;
  }

  modelUrlInput.blur();
});

confidenceInput.addEventListener('input', () => {
  updateThresholdLabels();
  scheduleThresholdPreview();
});

iouInput.addEventListener('input', () => {
  updateThresholdLabels();
  scheduleThresholdPreview();
});

runButton.addEventListener('click', () => {
  handleRunButtonClick().catch(error => {
    writeOutput(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

startCameraButton.addEventListener('click', () => {
  startCamera().catch(error => {
    writeOutput(error instanceof Error ? error.stack ?? error.message : String(error));
  });
});

stopCameraButton.addEventListener('click', () => {
  stopCamera();
});

updateInputMode();
updateRunButton();
updateThresholdLabels();
loadSelectedModel().catch(error => {
  writeModelInfo(error instanceof Error ? error.stack ?? error.message : String(error));
});

async function loadSelectedModel(): Promise<void> {
  const token = ++loadingModelToken;
  setLoadingModel(true);
  writeModelInfo('正在加载模型...');
  writeOutput('模型加载中，请稍候。');

  try {
    const model = await getModelSource();
    const labels = await getClassNames();
    const executionProvider = getSelectedExecutionProvider();
    const nextYolo = await Yolo.create({
      model,
      labels,
      wasmPaths: ORT_WASM_PATHS,
      //imageResize:'proportional',
      executionProviders: [executionProvider],
    });

    if (token !== loadingModelToken) {
      await nextYolo.dispose();
      return;
    }

    await yolo?.dispose();
    yolo = nextYolo;
    activeExecutionProviders = [executionProvider];

    writeModelInfo(formatModelInfo(yolo.onnxModel));
    writeOutput('模型已加载。请选择图片，然后点击运行推理。');
  } finally {
    if (token === loadingModelToken) {
      setLoadingModel(false);
    }
  }
}

async function handleRunButtonClick(): Promise<void> {
  if (getInputMode() === 'camera') {
    if (isCameraInferencing) {
      stopCameraInference();
      return;
    }

    startCameraInference();
    return;
  }

  await runImageInference();
}

async function runImageInference(): Promise<void> {
  if (!yolo) {
    throw new Error('请先选择模型文件或填写模型 URL，等待模型加载完成。');
  }

  const source = await getImageElement();

  setInferencing(true);

  try {
    writeOutput('正在推理...');

    const startedAt = performance.now();
    const result = await runInferenceByModelType(yolo, source);
    const elapsedMs = performance.now() - startedAt;

    drawInferenceResult(yolo, source, result, preview);
    writeOutput(formatInferenceResult(result, elapsedMs));
  } finally {
    setInferencing(false);
  }
}

async function runCurrentPreviewInference(): Promise<void> {
  if (!yolo) {
    return;
  }

  if (getInputMode() === 'camera') {
    if (!cameraStream || cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || isCameraInferencing) {
      return;
    }

    await runSingleSourceInference(cameraVideo, false);
    return;
  }

  if (!imageFileInput.files?.[0]) {
    return;
  }

  await runImageInference();
}

async function runSingleSourceInference(source: YoloImageSource, drawSource = true): Promise<void> {
  if (!yolo) {
    return;
  }

  setInferencing(true);

  try {
    writeOutput('正在按当前阈值重新推理...');

    const startedAt = performance.now();
    const result = await runInferenceByModelType(yolo, source);
    const elapsedMs = performance.now() - startedAt;

    drawInferenceResult(yolo, source, result, preview, drawSource);
    writeOutput(formatInferenceResult(result, elapsedMs));
  } finally {
    setInferencing(false);
  }
}

function startCameraInference(): void {
  if (!yolo) {
    throw new Error('请先选择模型文件或填写模型 URL，等待模型加载完成。');
  }

  if (!cameraStream || cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error('请先打开摄像头并等待画面出现。');
  }

  isCameraInferencing = true;
  resetFps();
  setInferencing(true);
  writeOutput('摄像头连续推理中...');
  cameraInferenceFrame = requestAnimationFrame(runCameraInferenceLoop);
}

function stopCameraInference(): void {
  isCameraInferencing = false;

  if (cameraInferenceFrame) {
    cancelAnimationFrame(cameraInferenceFrame);
    cameraInferenceFrame = 0;
  }

  setInferencing(false);
  resetFps();
  writeOutput('摄像头推理已停止。');
}

async function runCameraInferenceLoop(): Promise<void> {
  if (!isCameraInferencing || !yolo || !cameraStream) {
    stopCameraInference();
    return;
  }

  if (cameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    cameraInferenceFrame = requestAnimationFrame(runCameraInferenceLoop);
    return;
  }

  try {
    const startedAt = performance.now();
    const result = await runInferenceByModelType(yolo, cameraVideo);
    const elapsedMs = performance.now() - startedAt;

    drawInferenceResult(yolo, cameraVideo, result, preview, false);
    updateFps(performance.now());
    writeOutput(formatInferenceResult(result, elapsedMs));
  } catch (error) {
    stopCameraInference();
    writeOutput(error instanceof Error ? error.stack ?? error.message : String(error));
    return;
  }

  if (isCameraInferencing) {
    cameraInferenceFrame = requestAnimationFrame(runCameraInferenceLoop);
  }
}

async function getModelSource(): Promise<YoloModelSource> {
  const file = modelFileInput.files?.[0];

  if (file) {
    return new Uint8Array(await file.arrayBuffer());
  }

  const url = modelUrlInput.value.trim();
  return url || DEFAULT_MODEL_URL;
}

async function getClassNames(): Promise<string[] | undefined> {
  const file = classNamesFileInput.files?.[0];

  if (!file) {
    return undefined;
  }

  return parseClassNames(await file.text());
}

function parseClassNames(content: string): string[] {
  const trimmed = content.trim();

  if (!trimmed) {
    return [];
  }

  try {
    const parsed = JSON.parse(trimmed);

    if (Array.isArray(parsed)) {
      return parsed.map(String).map(label => label.trim()).filter(Boolean);
    }
  } catch {
    // class_names.txt normally contains one class per line.
  }

  return trimmed
    .split(/\r?\n|,/)
    .map(label => label.trim())
    .filter(Boolean);
}

async function getImageElement(): Promise<HTMLImageElement> {
  const file = imageFileInput.files?.[0];

  if (!file) {
    throw new Error('请选择一张图片。');
  }

  const url = URL.createObjectURL(file);

  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function startCamera(): Promise<void> {
  stopCamera();
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT },
      facingMode: 'environment',
    },
    audio: false,
  });
  cameraVideo.srcObject = cameraStream;
  await cameraVideo.play();
  syncCameraViewport();
  startCameraButton.disabled = true;
  stopCameraButton.disabled = false;
  writeOutput('摄像头已打开。点击运行推理开始连续检测。');
  updateRunButton();
}

function stopCamera(): void {
  stopCameraInference();
  cameraStream?.getTracks().forEach(track => track.stop());
  cameraStream = null;
  cameraVideo.pause();
  cameraVideo.srcObject = null;
  preview.getContext('2d')?.clearRect(0, 0, preview.width, preview.height);
  syncCameraViewport(CAMERA_WIDTH, CAMERA_HEIGHT);
  startCameraButton.disabled = false;
  stopCameraButton.disabled = true;
  resetFps();
  updateRunButton();
}

async function runInferenceByModelType(yoloInstance: Yolo, source: YoloImageSource): Promise<InferenceResult> {
  const modelType = yoloInstance.onnxModel.modelType;
  const confidence = getThresholdValue(confidenceInput, 0.2);
  const iou = getThresholdValue(iouInput, 0.7);

  switch (modelType) {
    case 'Classification':
      return {
        modelType,
        result: await yoloInstance.RunClassification(source),
      };
    case 'ObjectDetection':
      return {
        modelType,
        result: await yoloInstance.RunObjectDetection(source, confidence, iou),
      };
    case 'ObbDetection':
      return {
        modelType,
        result: await yoloInstance.RunObbDetection(source, confidence, iou),
      };
    case 'Segmentation':
      return {
        modelType,
        result: await yoloInstance.RunSegmentation(source, confidence, 0.65, iou),
      };
    case 'PoseEstimation':
      return {
        modelType,
        result: await yoloInstance.RunPoseEstimation(source, confidence, iou),
      };
    default:
      throw new Error(`Unsupported model type: ${modelType satisfies never}`);
  }
}

function getThresholdValue(input: HTMLInputElement, fallback: number): number {
  const value = Number(input.value);

  if (!Number.isFinite(value)) {
    return fallback;
  }

  return Math.min(Math.max(value, 0), 1);
}

function updateThresholdLabels(): void {
  confidenceValue.textContent = getThresholdValue(confidenceInput, 0.2).toFixed(2);
  iouValue.textContent = getThresholdValue(iouInput, 0.7).toFixed(2);
}

function scheduleThresholdPreview(): void {
  window.clearTimeout(thresholdPreviewTimer);
  thresholdPreviewTimer = window.setTimeout(() => {
    runThresholdPreview().catch(error => {
      writeOutput(error instanceof Error ? error.stack ?? error.message : String(error));
    });
  }, 150);
}

async function runThresholdPreview(): Promise<void> {
  if (isThresholdPreviewRunning) {
    pendingThresholdPreview = true;
    return;
  }

  isThresholdPreviewRunning = true;

  try {
    await runCurrentPreviewInference();
  } finally {
    isThresholdPreviewRunning = false;

    if (pendingThresholdPreview) {
      pendingThresholdPreview = false;
      scheduleThresholdPreview();
    }
  }
}

function drawInferenceResult(
  yoloInstance: Yolo,
  source: YoloImageSource,
  inference: InferenceResult,
  canvas: HTMLCanvasElement,
  drawSource = true,
): void {
  switch (inference.modelType) {
    case 'Classification':
      yoloInstance.drawClassifications(source, inference.result, canvas, { drawSource });
      break;
    case 'ObjectDetection':
      yoloInstance.drawObjectDetections(source, inference.result, canvas, { drawSource });
      break;
    case 'ObbDetection':
      yoloInstance.drawObbDetections(source, inference.result, canvas, { drawSource });
      break;
    case 'Segmentation':
      yoloInstance.drawSegmentations(source, inference.result, canvas, { drawSource });
      break;
    case 'PoseEstimation':
      yoloInstance.drawPoseEstimations(source, inference.result, canvas, { drawSource });
      break;
    default:
      throw new Error(`Unsupported model type: ${inference satisfies never}`);
  }
}

function formatInferenceResult(inference: InferenceResult, elapsedMs: number): string {
  switch (inference.modelType) {
    case 'Classification':
      return formatClassificationResult(inference.result, elapsedMs);
    case 'ObjectDetection':
      return formatDetectionResult(inference.modelType, inference.result, elapsedMs);
    case 'ObbDetection':
      return formatObbDetectionResult(inference.result, elapsedMs);
    case 'Segmentation':
      return formatSegmentationResult(inference.result, elapsedMs);
    case 'PoseEstimation':
      return formatPoseEstimationResult(inference.result, elapsedMs);
    default:
      throw new Error(`Unsupported model type: ${inference satisfies never}`);
  }
}

function formatClassificationResult(result: Classification[], elapsedMs: number): string {
  const lines = [`推理完成，耗时 ${elapsedMs.toFixed(2)} ms`, 'modelType: Classification', `分类数量: ${result.length}`];

  for (const [index, item] of result.entries()) {
    lines.push(`${index + 1}. ${item.label} confidence=${item.confidence.toFixed(6)}`);
  }

  return lines.join('\n');
}

function formatDetectionResult(modelType: ModelType, result: ObjectDetection[], elapsedMs: number): string {
  const lines = [`推理完成，耗时 ${elapsedMs.toFixed(2)} ms`, `modelType: ${modelType}`, `检测数量: ${result.length}`];

  for (const [index, detection] of result.entries()) {
    const { left, top, right, bottom } = detection.boundingBox;

    lines.push(
      `${index + 1}. ${detection.label.name} ` +
      `confidence=${detection.confidence.toFixed(6)} ` +
      `box=[${left}, ${top}, ${right}, ${bottom}]`,
    );
  }

  return lines.join('\n');
}

function formatObbDetectionResult(result: OBBDetection[], elapsedMs: number): string {
  const lines = [`推理完成，耗时 ${elapsedMs.toFixed(2)} ms`, 'modelType: ObbDetection', `检测数量: ${result.length}`];

  for (const [index, detection] of result.entries()) {
    const { left, top, right, bottom } = detection.boundingBox;

    lines.push(
      `${index + 1}. ${detection.label.name} ` +
      `confidence=${detection.confidence.toFixed(6)} ` +
      `angle=${detection.orientationAngle.toFixed(6)} ` +
      `box=[${left}, ${top}, ${right}, ${bottom}]`,
    );
  }

  return lines.join('\n');
}

function formatSegmentationResult(result: Segmentation[], elapsedMs: number): string {
  const lines = [`推理完成，耗时 ${elapsedMs.toFixed(2)} ms`, 'modelType: Segmentation', `分割数量: ${result.length}`];

  for (const [index, detection] of result.entries()) {
    const { left, top, right, bottom } = detection.boundingBox;

    lines.push(
      `${index + 1}. ${detection.label.name} ` +
      `confidence=${detection.confidence.toFixed(6)} ` +
      `maskBytes=${detection.bitPackedPixelMask.byteLength} ` +
      `box=[${left}, ${top}, ${right}, ${bottom}]`,
    );
  }

  return lines.join('\n');
}

function formatPoseEstimationResult(result: PoseEstimation[], elapsedMs: number): string {
  const lines = [`推理完成，耗时 ${elapsedMs.toFixed(2)} ms`, 'modelType: PoseEstimation', `姿态数量: ${result.length}`];

  for (const [index, detection] of result.entries()) {
    const { left, top, right, bottom } = detection.boundingBox;

    lines.push(
      `${index + 1}. ${detection.label.name} ` +
      `confidence=${detection.confidence.toFixed(6)} ` +
      `keyPoints=${detection.keyPoints.length} ` +
      `box=[${left}, ${top}, ${right}, ${bottom}]`,
    );
  }

  return lines.join('\n');
}

function formatModelInfo(model: OnnxModel): string {
  return [
    '模型加载完成',
    `executionProviders: ${activeExecutionProviders.join(' + ')}`,
    `modelType: ${model.modelType}`,
    `modelVersion: ${model.modelVersion}`,
    `modelDataType: ${model.modelDataType}`,
    `inputShapes: ${JSON.stringify(model.inputShapes)}`,
    `outputShapes: ${JSON.stringify(model.outputShapes)}`,
    `inputShapeSize: ${model.inputShapeSize}`,
    `labels: ${model.labels.slice(0, 10).map(label => `${label.index}:${label.name}`).join(', ')}${model.labels.length > 10 ? '...' : ''}`,
  ].join('\n');
}

function setLoadingModel(isLoading: boolean): void {
  backendSelect.disabled = isLoading;
  inputModeSelect.disabled = isLoading;
  modelFileInput.disabled = isLoading;
  classNamesFileInput.disabled = isLoading;
  modelUrlInput.disabled = isLoading;
  updateRunButton(isLoading);
}

function setInferencing(isInferencing: boolean): void {
  updateRunButton(false, isInferencing);
}

function updateRunButton(isLoadingModel = false, isInferencing = false): void {
  const inputReady =
    getInputMode() === 'camera'
      ? Boolean(cameraStream && cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)
      : Boolean(imageFileInput.files?.[0]);

  runButton.disabled = isLoadingModel || (!isCameraInferencing && (isInferencing || !yolo || !inputReady));
  runButton.textContent = isCameraInferencing ? '停止推理' : (isInferencing ? '推理中...' : '运行推理');
}

function writeOutput(message: string): void {
  output.textContent = message;
}

function writeModelInfo(message: string): void {
  modelInfo.textContent = message;
}

function updateFps(now: number): void {
  if (fpsLastTimestamp === 0) {
    fpsLastTimestamp = now;
    fpsBadge.textContent = 'FPS: --';
    return;
  }

  const delta = now - fpsLastTimestamp;
  fpsLastTimestamp = now;

  if (delta <= 0) {
    return;
  }

  const currentFps = 1000 / delta;
  fpsAverage = fpsAverage === 0 ? currentFps : fpsAverage * 0.85 + currentFps * 0.15;
  fpsBadge.textContent = `FPS: ${fpsAverage.toFixed(1)}`;
}

function resetFps(): void {
  fpsLastTimestamp = 0;
  fpsAverage = 0;
  fpsBadge.textContent = 'FPS: --';
}

function syncCameraViewport(width = cameraVideo.videoWidth || CAMERA_WIDTH, height = cameraVideo.videoHeight || CAMERA_HEIGHT): void {
  cameraVideo.width = width;
  cameraVideo.height = height;
  preview.width = width;
  preview.height = height;
  viewer.style.aspectRatio = `${width} / ${height}`;
}

function hasModelSource(): boolean {
  return Boolean(modelFileInput.files?.[0] || modelUrlInput.value.trim() || DEFAULT_MODEL_URL);
}

function getSelectedExecutionProvider(): YoloExecutionProvider {
  return backendSelect.value as YoloExecutionProvider;
}

function renderBackendOptions(): void {
  backendSelect.replaceChildren(
    ...YoloWebExecutionProviderOptions.map(({ value, label }, index) => {
      const option = document.createElement('option');
      option.value = String(value);
      option.textContent = label;
      option.selected = index === 0;

      return option;
    }),
  );
}

function getInputMode(): 'image' | 'camera' {
  return inputModeSelect.value === 'camera' ? 'camera' : 'image';
}

function updateInputMode(): void {
  const isCamera = getInputMode() === 'camera';

  viewer.classList.toggle('camera-mode', isCamera);
  imageInputGroup.style.display = isCamera ? 'none' : '';
  startCameraButton.style.display = isCamera ? '' : 'none';
  stopCameraButton.style.display = isCamera ? '' : 'none';

  if (!isCamera) {
    stopCamera();
    writeOutput('图片模式：请选择图片并点击运行推理。');
  } else {
    writeOutput('摄像头模式：请先打开摄像头，然后点击运行推理开始连续检测。');
  }

  updateRunButton();
}

function getElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }

  return element;
}

