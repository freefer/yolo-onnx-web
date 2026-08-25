/// <reference path="./vite-env.d.ts" />
import {
  canReuseOrtBundle,
  getLoadedOrtBundle,
  resolveOrtBundle,
  Sam3,
  splitTextPrompts,
  YoloWebExecutionProviderOptions,
} from '../../src';
import type { Point, Rect, Segmentation, SegmentationDrawingOptions, YoloExecutionProvider, YoloModelSource } from '../../src';

const DEMO_MODE = import.meta.env.YOLO_DEMO_MODE === 'pages' ? 'pages' : 'local';
const ORT_SOURCE = import.meta.env.YOLO_ORT_SOURCE === 'cdn' ? 'cdn' : 'npm';
const ORT_WASM_PATHS =
  import.meta.env.YOLO_ORT_WASM_PATHS ||
  new URL('./ort-wasm/', window.location.href).toString();
const BACKEND_STORAGE_KEY = 'yolo-onnx-web:backend';
const DRAG_THRESHOLD_PX = 6;

const backendSelect = getElement<HTMLSelectElement>('#backend');
const taskModeSelect = getElement<HTMLSelectElement>('#taskMode');
const promptPolaritySelect = getElement<HTMLSelectElement>('#promptPolarity');
const textPromptInput = getElement<HTMLInputElement>('#textPrompt');
const textDescriptionInput = getElement<HTMLInputElement>('#textDescription');
const modelDirectoryInput = getElement<HTMLInputElement>('#modelDirectory');
const visionFileInput = getElement<HTMLInputElement>('#visionFile');
const textFileInput = getElement<HTMLInputElement>('#textFile');
const groundingFileInput = getElement<HTMLInputElement>('#groundingFile');
const promptFileInput = getElement<HTMLInputElement>('#promptFile');
const tokenizerFileInput = getElement<HTMLInputElement>('#tokenizerFile');
const confidenceInput = getElement<HTMLInputElement>('#confidenceInput');
const resultOpacityInput = getElement<HTMLInputElement>('#resultOpacityInput');
const confidenceValue = getElement<HTMLElement>('#confidenceValue');
const resultOpacityValue = getElement<HTMLElement>('#resultOpacityValue');
const imageFileInput = getElement<HTMLInputElement>('#imageFile');
const loadButton = getElement<HTMLButtonElement>('#loadButton');
const encodeButton = getElement<HTMLButtonElement>('#encodeButton');
const runTextButton = getElement<HTMLButtonElement>('#runTextButton');
const resetButton = getElement<HTMLButtonElement>('#resetButton');
const candidates = getElement<HTMLElement>('#candidates');
const hint = getElement<HTMLElement>('#hint');
const preview = getElement<HTMLCanvasElement>('#preview');
const modelInfo = getElement<HTMLPreElement>('#modelInfo');
const output = getElement<HTMLPreElement>('#output');
const loadingOverlay = getElement<HTMLElement>('#loadingOverlay');
const loadingTitle = getElement<HTMLElement>('#loadingTitle');
const loadingMessage = getElement<HTMLElement>('#loadingMessage');
const demoTitle = getElement<HTMLElement>('#demoTitle');
const demoDescription = getElement<HTMLElement>('#demoDescription');
const demoModeBadge = getElement<HTMLElement>('#demoModeBadge');

type TaskMode = 'pcs' | 'pvs';
type PromptPolarity = 'positive' | 'negative';

let sam3: Sam3 | null = null;
let sourceImage: HTMLImageElement | null = null;
let currentMasks: Segmentation[] = [];
let lastPvsMasks: Segmentation[] = [];
let lastPvsLowRes: Array<{ logits: Float32Array; width: number; height: number }> = [];
let dragStart: Point | null = null;
let dragCurrent: Point | null = null;
let pendingPrompt: { box?: Rect; point?: Point; positive: boolean } | null = null;
let isBusy = false;
let blobUrls: string[] = [];

applyDemoModeUi();
renderBackendOptions();
updateThresholdLabels();
updateHint();
updateButtons();

backendSelect.addEventListener('change', () => {
  sessionStorage.setItem(BACKEND_STORAGE_KEY, backendSelect.value);
  reloadPageIfOrtBundleIncompatible();
});

taskModeSelect.addEventListener('change', () => {
  updateHint();
  redraw();
});

confidenceInput.addEventListener('input', () => {
  updateThresholdLabels();
});

confidenceInput.addEventListener('change', () => {
  applyConfidence().catch(error => writeOutput(formatError(error)));
});

resultOpacityInput.addEventListener('input', () => {
  updateThresholdLabels();
  redraw();
});

loadButton.addEventListener('click', () => {
  loadModels().catch(error => writeOutput(formatError(error)));
});

encodeButton.addEventListener('click', () => {
  encodeImage().catch(error => writeOutput(formatError(error)));
});

runTextButton.addEventListener('click', () => {
  runTextPrompt().catch(error => writeOutput(formatError(error)));
});

resetButton.addEventListener('click', () => {
  resetPrompts();
});

textPromptInput.addEventListener('keydown', event => {
  if (event.key === 'Enter') {
    event.preventDefault();
    runTextPrompt().catch(error => writeOutput(formatError(error)));
  }
});

textDescriptionInput.addEventListener('keydown', event => {
  if (event.key === 'Enter') {
    event.preventDefault();
    runTextPrompt().catch(error => writeOutput(formatError(error)));
  }
});

modelDirectoryInput.addEventListener('change', () => {
  applyLocalModelDirectory();
});

for (const slot of modelSlots()) {
  slot.input.addEventListener('change', () => {
    const file = slot.input.files?.[0];
    writeOutput(file ? `已选择${slot.name}: ${localFileLabel(file)}` : `已清除${slot.name}。`);
  });
}

imageFileInput.addEventListener('change', () => {
  loadImageFile().catch(error => writeOutput(formatError(error)));
});

preview.addEventListener('pointerdown', event => {
  handlePointerDown(event);
});

preview.addEventListener('pointermove', event => {
  handlePointerMove(event);
});

window.addEventListener('pointerup', event => {
  handlePointerUp(event).catch(error => writeOutput(formatError(error)));
});

preview.addEventListener('contextmenu', event => {
  event.preventDefault();
});

async function loadModels(): Promise<void> {
  if (reloadPageIfOrtBundleIncompatible()) {
    return;
  }

  await setBusy(true, '正在加载 SAM3 模型，体积较大请稍候...');
  writeModelInfo('正在加载 SAM3 模型...');

  try {
    const sources = await resolveModelSources();
    const executionProvider = getSelectedExecutionProvider();
    const next = await Sam3.create({
      visionEncoder: sources.vision,
      textEncoder: sources.text,
      groundingDecoder: sources.grounding,
      promptDecoder: sources.prompt,
      tokenizer: sources.tokenizer,
      wasmPaths: ORT_WASM_PATHS,
      executionProviders: [executionProvider],
      numThreads: 1,
      confidenceThreshold: getThresholdValue(confidenceInput, 0.5),
      onLoadProgress: message => {
        void setBusy(true, message);
      },
    });

    await sam3?.dispose();
    sam3 = next;
    sourceImage = null;
    currentMasks = [];
    lastPvsMasks = [];
    lastPvsLowRes = [];
    clearCandidates();

    writeModelInfo(
      [
        'SAM3 模型已加载',
        `executionProviders: ${executionProvider}`,
        `ortSource: ${ORT_SOURCE}`,
        'sessions: vision / text / grounding / prompt',
        'tokenizer: clip_bpe.json',
      ].join('\n'),
    );
    writeOutput('模型已加载。请选择图片，系统会先编码图像特征。');
  } catch (error) {
    if (isOrtBundleReloadError(error)) {
      reloadPageIfOrtBundleIncompatible(true);
      return;
    }

    throw error;
  } finally {
    await setBusy(false);
  }
}

async function loadImageFile(): Promise<void> {
  const file = imageFileInput.files?.[0];

  if (!file) {
    return;
  }

  sourceImage = await decodeImageFile(file);
  currentMasks = [];
  lastPvsMasks = [];
  lastPvsLowRes = [];
  clearCandidates();
  redraw();
  writeOutput(`已选择图片 ${sourceImage.naturalWidth}×${sourceImage.naturalHeight}。`);

  if (sam3) {
    await encodeImage();
  }
}

async function encodeImage(): Promise<void> {
  if (!sam3) {
    throw new Error('请先加载 SAM3 模型。');
  }

  if (!sourceImage) {
    throw new Error('请先选择一张图片。');
  }

  await setBusy(true, '正在编码图像特征...');
  const startedAt = performance.now();

  try {
    await sam3.setImage(sourceImage);
    currentMasks = [];
    lastPvsMasks = [];
    lastPvsLowRes = [];
    clearCandidates();
    redraw();
    writeOutput(`图像已编码，耗时 ${(performance.now() - startedAt).toFixed(0)} ms。现在可以输入文本或在图上点选。`);
  } finally {
    await setBusy(false);
  }
}

async function runTextPrompt(): Promise<void> {
  if (!sam3?.inferenceState) {
    throw new Error('请先编码图像。');
  }

  const prompt = requireTextPrompt();
  const description = getTextDescription();
  const texts = splitTextPrompts(prompt);
  await setBusy(
    true,
    texts.length > 1
      ? `正在按 ${texts.length} 个类别做概念分割（${texts.join('、')}）...`
      : `正在按类别 “${prompt}” 做概念分割...`,
  );
  const startedAt = performance.now();

  try {
    taskModeSelect.value = 'pcs';
    currentMasks = await sam3.setTextPrompt(prompt, description);
    lastPvsMasks = [];
    lastPvsLowRes = [];
    clearCandidates();
    redraw();
    writeOutput(formatMaskResult('PCS 文本分割', currentMasks, performance.now() - startedAt));
  } finally {
    await setBusy(false);
  }
}

async function applyConfidence(): Promise<void> {
  if (!sam3?.inferenceState?.textEmbeddings) {
    return;
  }

  await setBusy(true, '正在按新阈值过滤概念分割结果...');

  try {
    const result = await sam3.setConfidenceThreshold(getThresholdValue(confidenceInput, 0.5));
    if (Array.isArray(result)) {
      currentMasks = result;
      redraw();
      writeOutput(formatMaskResult('PCS 阈值更新', currentMasks, 0));
    }
  } finally {
    await setBusy(false);
  }
}

function resetPrompts(): void {
  if (!sam3?.inferenceState) {
    return;
  }

  if (getTaskMode() === 'pvs') {
    sam3.resetVisualPrompts();
    lastPvsMasks = [];
    lastPvsLowRes = [];
    clearCandidates();
  } else {
    sam3.resetPrompts();
  }

  currentMasks = [];
  redraw();
  writeOutput(getTaskMode() === 'pvs' ? '已清空 PVS 点/框/掩码。' : '已清空 PCS 文本与范例。');
}

function handlePointerDown(event: PointerEvent): void {
  if (isBusy || !sam3?.inferenceState || !sourceImage) {
    return;
  }

  preview.setPointerCapture(event.pointerId);
  dragStart = eventToImagePoint(event);
  dragCurrent = dragStart;
}

function handlePointerMove(event: PointerEvent): void {
  if (!dragStart) {
    return;
  }

  dragCurrent = eventToImagePoint(event);
  redraw();
}

async function handlePointerUp(event: PointerEvent): Promise<void> {
  if (!dragStart || !sam3?.inferenceState) {
    dragStart = null;
    dragCurrent = null;
    return;
  }

  const end = eventToImagePoint(event);
  const start = dragStart;
  dragStart = null;
  dragCurrent = null;

  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const isBox = Math.hypot(dx, dy) >= DRAG_THRESHOLD_PX;
  const negative = event.button === 2 || getPromptPolarity() === 'negative';
  pendingPrompt = isBox
    ? { box: normalizeRect(start, end), positive: !negative }
    : { point: end, positive: !negative };
  redraw();

  try {
    if (getTaskMode() === 'pcs') {
      await runPcsInteraction(start, end, isBox, !negative);
      return;
    }

    await runPvsInteraction(start, end, isBox, negative);
  } finally {
    pendingPrompt = null;
    redraw();
  }
}

async function runPcsInteraction(start: Point, end: Point, isBox: boolean, positive: boolean): Promise<void> {
  if (!sam3) {
    return;
  }

  const prompt = requireTextPrompt();
  const description = getTextDescription();
  const busyTitle = isBox ? '正在加入框范例' : '正在加入点范例';
  writeOutput(`${busyTitle}...`);
  await setBusy(true, busyTitle);
  const startedAt = performance.now();

  try {
    if (sam3.inferenceState?.text !== prompt || sam3.inferenceState?.textDescription !== description) {
      await sam3.setTextPrompt(prompt, description);
    }

    if (isBox) {
      currentMasks = await sam3.addGeometricPrompt(normalizeRect(start, end), positive);
    } else {
      currentMasks = await sam3.addGeometricPoint(start, positive);
    }

    redraw();
    writeOutput(formatMaskResult(positive ? 'PCS 正例修正' : 'PCS 负例修正', currentMasks, performance.now() - startedAt));
  } finally {
    await setBusy(false);
  }
}

async function runPvsInteraction(start: Point, end: Point, isBox: boolean, negative: boolean): Promise<void> {
  if (!sam3) {
    return;
  }

  const busyTitle = isBox ? '正在按框分割实例' : `正在加入${negative ? '背景' : '前景'}点`;
  writeOutput(`${busyTitle}...`);
  await setBusy(true, busyTitle);
  const startedAt = performance.now();

  try {
    const result = isBox
      ? await sam3.addBox(normalizeRect(start, end))
      : await sam3.addPoint(start, negative ? 0 : 1);

    lastPvsMasks = result.masks;
    const maskSize = sam3.inferenceState?.pvsMaskInput;
    lastPvsLowRes = result.lowResMasks.map(logits => ({
      logits,
      width: maskSize?.width ?? 288,
      height: maskSize?.height ?? 288,
    }));
    const bestIndex = argmax(result.ious);
    currentMasks = [result.masks[bestIndex] ?? result.masks[0]].filter(Boolean);
    renderCandidates(result.ious, bestIndex);
    redraw();
    writeOutput(
      [
        formatMaskResult('PVS 实例分割', currentMasks, performance.now() - startedAt),
        `iou=${result.ious.map(value => value.toFixed(3)).join(', ')}`,
        `objectScore=${result.objectScores.map(value => value.toFixed(3)).join(', ')}`,
      ].join('\n'),
    );
  } finally {
    await setBusy(false);
  }
}

async function usePvsCandidate(index: number): Promise<void> {
  const mask = lastPvsMasks[index];
  const lowRes = lastPvsLowRes[index];

  if (!sam3 || !mask || !lowRes) {
    return;
  }

  currentMasks = [mask];
  await setBusy(true, `正在锁定候选 ${index + 1}`);

  try {
    const result = await sam3.addMask(lowRes);
    const bestIndex = argmax(result.ious);
    currentMasks = [result.masks[bestIndex] ?? result.masks[0]].filter(Boolean);
    lastPvsMasks = result.masks;
    lastPvsLowRes = result.lowResMasks.map(logits => ({
      logits,
      width: lowRes.width,
      height: lowRes.height,
    }));
    redraw();
    writeOutput(`已选择候选 ${index + 1}。继续点击可在此掩码上精细化。`);
  } finally {
    await setBusy(false);
  }
}

function renderCandidates(ious: readonly number[], selectedIndex: number): void {
  candidates.replaceChildren();

  if (ious.length <= 1) {
    return;
  }

  for (let index = 0; index < ious.length; index += 1) {
    const button = document.createElement('button');
    button.type = 'button';
    button.disabled = index === selectedIndex;
    button.textContent = `候选 ${index + 1}  IoU ${ious[index]?.toFixed(3) ?? '--'}`;
    button.addEventListener('click', () => {
      usePvsCandidate(index).catch(error => writeOutput(formatError(error)));
    });
    candidates.append(button);
  }
}

function clearCandidates(): void {
  candidates.replaceChildren();
}

function redraw(): void {
  if (!sourceImage) {
    return;
  }

  const context = preview.getContext('2d');

  if (!context) {
    return;
  }

  if (sam3 && currentMasks.length > 0) {
    sam3.drawSegmentationEdgePoints(sourceImage, currentMasks, preview, getSegmentationDrawOptions());
  } else {
    preview.width = sourceImage.naturalWidth;
    preview.height = sourceImage.naturalHeight;
    context.drawImage(sourceImage, 0, 0);
  }

  drawPromptOverlay(context);
}

function drawPromptOverlay(context: CanvasRenderingContext2D): void {
  if (dragStart && dragCurrent) {
    strokeRect(context, normalizeRect(dragStart, dragCurrent), '#f8fafc');
  }

  if (pendingPrompt?.box) {
    strokeRect(context, pendingPrompt.box, pendingPrompt.positive ? '#22c55e' : '#ef4444');
  } else if (pendingPrompt?.point) {
    drawPointMarker(context, pendingPrompt.point, pendingPrompt.positive);
  }

  if (currentMasks.length > 0) {
    return;
  }

  const state = sam3?.inferenceState;

  if (!state) {
    return;
  }

  if (getTaskMode() === 'pcs') {
    for (const item of state.boxes) {
      strokeRect(context, item.box, item.label ? '#22c55e' : '#ef4444');
    }

    for (const item of state.points) {
      drawPointMarker(context, item.point, item.label > 0);
    }
    return;
  }

  if (state.pvsBox) {
    strokeRect(context, state.pvsBox, '#38bdf8');
  }

  for (const item of state.pvsPoints) {
    drawPointMarker(context, item.point, item.label === 1);
  }
}

function strokeRect(context: CanvasRenderingContext2D, box: Rect, color: string): void {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, preview.width / 400);
  context.setLineDash([8, 4]);
  context.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
  context.restore();
}

function drawPointMarker(context: CanvasRenderingContext2D, point: Point, positive: boolean): void {
  const radius = Math.max(5, preview.width / 180);
  context.save();
  context.fillStyle = positive ? '#22c55e' : '#ef4444';
  context.strokeStyle = '#0f172a';
  context.lineWidth = 2;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.restore();
}

async function resolveModelSources(): Promise<{
  vision: YoloModelSource;
  text: YoloModelSource;
  grounding: YoloModelSource;
  prompt: YoloModelSource;
  tokenizer: string;
}> {
  revokeBlobUrls();
  const directoryFiles = filesFromDirectory(modelDirectoryInput.files);

  return {
    vision: sourceFromFile(resolveSlotFile(visionFileInput, directoryFiles, ['vision-encoder.fp16.onnx', 'vision-encoder.onnx']), '视觉编码器'),
    text: sourceFromFile(resolveSlotFile(textFileInput, directoryFiles, ['text-encoder.fp16.onnx', 'text-encoder.onnx']), '文本编码器'),
    grounding: sourceFromFile(resolveSlotFile(groundingFileInput, directoryFiles, ['grounding-decoder.fp16.onnx', 'grounding-decoder.onnx']), 'Grounding 解码器'),
    prompt: sourceFromFile(resolveSlotFile(promptFileInput, directoryFiles, ['prompt-decoder.fp16.onnx', 'prompt-decoder.onnx']), 'Prompt 解码器'),
    tokenizer: sourceFromFile(resolveSlotFile(tokenizerFileInput, directoryFiles, ['clip_bpe.json']), 'Tokenizer'),
  };
}

function modelSlots(): Array<{ name: string; input: HTMLInputElement; names: readonly string[] }> {
  return [
    { name: '视觉编码器', input: visionFileInput, names: ['vision-encoder.fp16.onnx', 'vision-encoder.onnx'] },
    { name: '文本编码器', input: textFileInput, names: ['text-encoder.fp16.onnx', 'text-encoder.onnx'] },
    { name: 'Grounding 解码器', input: groundingFileInput, names: ['grounding-decoder.fp16.onnx', 'grounding-decoder.onnx'] },
    { name: 'Prompt 解码器', input: promptFileInput, names: ['prompt-decoder.fp16.onnx', 'prompt-decoder.onnx'] },
    { name: 'Tokenizer', input: tokenizerFileInput, names: ['clip_bpe.json'] },
  ];
}

function pickModelFile(files: Map<string, File>, names: readonly string[]): File | undefined {
  for (const name of names) {
    const file = files.get(name);
    if (file) {
      return file;
    }
  }

  return undefined;
}

function filesFromDirectory(fileList: FileList | null): Map<string, File> {
  const files = new Map<string, File>();

  if (!fileList) {
    return files;
  }

  for (const file of fileList) {
    files.set(file.name.toLowerCase(), file);
  }

  return files;
}

function applyLocalModelDirectory(): void {
  const files = filesFromDirectory(modelDirectoryInput.files);

  if (files.size === 0) {
    for (const slot of modelSlots()) {
      assignFileToInput(slot.input, undefined);
    }
    writeOutput('未选到本地模型文件。');
    return;
  }

  const found: string[] = [];
  const missing: string[] = [];

  for (const slot of modelSlots()) {
    const file = pickModelFile(files, slot.names);
    assignFileToInput(slot.input, file);

    if (!file) {
      missing.push(slot.name);
      continue;
    }

    found.push(`${slot.name}: ${localFileLabel(file)}`);
  }

  writeOutput(
    [
      `已选择本地目录，共 ${files.size} 个文件。也可再单独替换某一个模型文件。`,
      ...found,
      missing.length > 0 ? `未找到: ${missing.join('、')}，请手动选择这些文件。` : '',
    ]
      .filter(Boolean)
      .join('\n'),
  );
}

function assignFileToInput(input: HTMLInputElement, file: File | undefined): void {
  if (!file) {
    input.value = '';
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
}

function resolveSlotFile(input: HTMLInputElement, directoryFiles: Map<string, File>, names: readonly string[]): File | undefined {
  return input.files?.[0] ?? pickModelFile(directoryFiles, names);
}

function localFileLabel(file: File): string {
  const nativePath = (file as File & { path?: string }).path?.trim();
  const relative = nativePath || (file.webkitRelativePath || '').trim() || file.name;
  return /Win/i.test(navigator.userAgent) ? relative.replace(/\//g, '\\') : relative;
}

function sourceFromFile(file: File | undefined, label: string): string {
  if (!file) {
    throw new Error(`缺少${label}。请选择 SAM3 模型目录，或手动选择该文件。`);
  }

  const blobUrl = URL.createObjectURL(file);
  blobUrls.push(blobUrl);
  return blobUrl;
}

function revokeBlobUrls(): void {
  for (const url of blobUrls) {
    URL.revokeObjectURL(url);
  }

  blobUrls = [];
}

async function decodeImageFile(file: File): Promise<HTMLImageElement> {
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

function eventToImagePoint(event: PointerEvent): Point {
  const rect = preview.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * preview.width,
    y: ((event.clientY - rect.top) / rect.height) * preview.height,
  };
}

function normalizeRect(start: Point, end: Point): Rect {
  return {
    left: Math.min(start.x, end.x),
    top: Math.min(start.y, end.y),
    right: Math.max(start.x, end.x),
    bottom: Math.max(start.y, end.y),
  };
}

function getSegmentationDrawOptions(): SegmentationDrawingOptions {
  return {
    drawSource: true,
    drawBoundingBoxes: true,
    drawLabel: true,
    drawSegmentationPixelMask: true,
    fillSegmentationEdgePoints: true,
    contourThickness: 2,
    resultOpacity: getThresholdValue(resultOpacityInput, 1),
  };
}

function requireTextPrompt(): string {
  const prompt = textPromptInput.value.trim();

  if (!prompt) {
    textPromptInput.focus();
    throw new Error('请输入类别名称，多个类别用逗号分隔，例如 person, car, dog。');
  }

  return prompt;
}

function getTextDescription(): string | undefined {
  const description = textDescriptionInput.value.trim();
  return description || undefined;
}

function formatMaskResult(title: string, masks: readonly Segmentation[], elapsedMs: number): string {
  const lines = [
    `${title}${elapsedMs > 0 ? `，耗时 ${elapsedMs.toFixed(0)} ms` : ''}`,
    `实例数量: ${masks.length}`,
  ];

  for (const [index, mask] of masks.entries()) {
    const { left, top, right, bottom } = mask.boundingBox;
    lines.push(
      `${index + 1}. ${mask.label.name} confidence=${mask.confidence.toFixed(3)} box=[${Math.round(left)}, ${Math.round(top)}, ${Math.round(right)}, ${Math.round(bottom)}] edgePoints=${mask.segmentationEdgePoints?.length ?? 0}`,
    );
  }

  return lines.join('\n');
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.stack ?? error.message : String(error);
}

function applyDemoModeUi(): void {
  const isPages = DEMO_MODE === 'pages';
  demoModeBadge.dataset.mode = DEMO_MODE;
  demoModeBadge.textContent = isPages ? '线上 CDN' : '本地 npm';
  demoTitle.textContent = isPages ? 'SAM3 在线 Demo' : 'SAM3 交互分割示例';
  demoDescription.textContent = isPages
    ? 'GitHub Pages 不托管 SAM3 权重。请选择本地导出目录（优先 *.fp16.onnx），或分别选择视觉/文本/Grounding/Prompt 与 clip_bpe.json。类别用逗号分隔。'
    : '请选择 SAM3 模型目录，或分别选择各个 ONNX 与 tokenizer。浏览器请优先 fp16 视觉/文本编码器与 WebGPU。类别用逗号分隔。';
  document.title = isPages ? 'SAM3 Online Demo' : 'SAM3 browser example';
}

function renderBackendOptions(): void {
  const savedBackend = sessionStorage.getItem(BACKEND_STORAGE_KEY);

  backendSelect.replaceChildren(
    ...YoloWebExecutionProviderOptions.map(({ value, label }, index) => {
      const option = document.createElement('option');
      option.value = String(value);
      option.textContent = label;
      option.selected = savedBackend ? savedBackend === String(value) : index === 0;
      return option;
    }),
  );
}

function getSelectedExecutionProvider(): YoloExecutionProvider {
  return backendSelect.value as YoloExecutionProvider;
}

function getTaskMode(): TaskMode {
  return taskModeSelect.value === 'pvs' ? 'pvs' : 'pcs';
}

function getPromptPolarity(): PromptPolarity {
  return promptPolaritySelect.value === 'negative' ? 'negative' : 'positive';
}

function getThresholdValue(input: HTMLInputElement, fallback: number): number {
  const value = Number(input.value);
  return Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : fallback;
}

function updateThresholdLabels(): void {
  confidenceValue.textContent = getThresholdValue(confidenceInput, 0.5).toFixed(2);
  resultOpacityValue.textContent = getThresholdValue(resultOpacityInput, 1).toFixed(2);
}

function updateHint(): void {
  hint.textContent =
    getTaskMode() === 'pcs'
      ? 'PCS：类别用逗号分隔，例如 person, car, dog。详细描述可选，用来补充颜色、姿态、场景；查询时会加到每个类别上，标签仍用类别名。'
      : 'PVS：左键加点，拖拽画框，只分割当前点/框里的那一件；右键或极性选“负例”添加背景点。';
}

function yieldToUi(): Promise<void> {
  return new Promise(resolve => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.setTimeout(resolve, 0);
      });
    });
  });
}

async function setBusy(busy: boolean, message = '处理中...'): Promise<void> {
  const shouldPaint = busy && !loadingOverlay.classList.contains('visible');
  isBusy = busy;
  loadingTitle.textContent = message;
  loadingMessage.textContent = busy ? '请稍候，浏览器端推理可能需要数秒。' : '';
  loadingOverlay.classList.toggle('visible', busy);
  loadingOverlay.setAttribute('aria-busy', String(busy));
  updateButtons();

  if (shouldPaint) {
    await yieldToUi();
  }
}

function updateButtons(): void {
  const hasImage = Boolean(sourceImage);
  const hasState = Boolean(sam3?.inferenceState);
  loadButton.disabled = isBusy;
  encodeButton.disabled = isBusy || !sam3 || !hasImage;
  runTextButton.disabled = isBusy || !hasState;
  resetButton.disabled = isBusy || !hasState;
  backendSelect.disabled = isBusy;
  imageFileInput.disabled = isBusy;
  modelDirectoryInput.disabled = isBusy;

  for (const slot of modelSlots()) {
    slot.input.disabled = isBusy;
  }
}

function writeOutput(message: string): void {
  output.textContent = message;
}

function writeModelInfo(message: string): void {
  modelInfo.textContent = message;
}

function reloadPageIfOrtBundleIncompatible(force = false): boolean {
  const loaded = getLoadedOrtBundle();
  const requested = resolveOrtBundle([getSelectedExecutionProvider()]);

  if (!force && (!loaded || canReuseOrtBundle(loaded, requested))) {
    return false;
  }

  sessionStorage.setItem(BACKEND_STORAGE_KEY, backendSelect.value);
  void setBusy(true, `正在切换运行时包 (${loaded ?? 'none'} → ${requested})`);
  window.location.reload();
  return true;
}

function isOrtBundleReloadError(error: unknown): boolean {
  return Boolean(
    error &&
      typeof error === 'object' &&
      'code' in error &&
      (error as { code?: string }).code === 'ORT_BUNDLE_RELOAD_REQUIRED',
  );
}

function argmax(values: readonly number[]): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index += 1) {
    if ((values[index] ?? Number.NEGATIVE_INFINITY) > bestValue) {
      bestValue = values[index] ?? Number.NEGATIVE_INFINITY;
      bestIndex = index;
    }
  }

  return bestIndex;
}

function getElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }

  return element;
}
