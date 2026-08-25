# yolo-onnx-web

基于 `onnxruntime-web` 的浏览器端 YOLO 与 SAM 3 推理库。

本库可以在浏览器中加载 Ultralytics 风格的 ONNX 模型，解析模型元数据，根据模型版本和任务类型自动选择输出解析器，并提供分类、检测、分割、姿态估计、旋转框等任务的 Canvas 绘制方法。同时提供 `Sam3`，用于 SAM 3.1 multiplex 的文本概念分割（PCS）和交互式视觉分割（PVS）。

GitHub 仓库：[https://github.com/freefer/yolo-onnx-web](https://github.com/freefer/yolo-onnx-web)

在线 Demo：[YOLO](https://freefer.github.io/yolo-onnx-web/examples/browser/) · [SAM 3](https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html)

English documentation: [README.md](https://github.com/freefer/yolo-onnx-web/blob/main/README.md)

## 功能特性

- 在浏览器中运行 ONNX 模型。
- 支持 WebGPU、WASM、WebNN、WebGL、CPU 等执行后端，具体取决于浏览器和设备支持。
- 自动解析 ONNX 自定义元数据：`task`、`description`、`names`。
- 支持图片、Canvas、视频、摄像头等 `CanvasImageSource` 输入。
- 提供高层任务 API：
  - 分类
  - 目标检测
  - 旋转框检测
  - 实例分割
  - 姿态估计
- 为所有支持的任务提供通用绘制方法。
- 支持 RF-DETR 目标检测和实例分割模型。
- 提供 `Sam3`，用于 SAM 3.1 multiplex：文本概念分割（PCS）和交互式视觉分割（PVS）。
- 导出 `DrawTool`，可以脱离 `Yolo` 实例单独使用绘制工具。

## 安装

```bash
npm install yolo-onnx-web
```

如果从当前仓库开发：

```bash
npm install
npm run build
npm start
```

浏览器示例使用 `vite.config.ts` 中配置的固定端口启动。

## 浏览器运行时配置

当前包依赖的 ONNX Runtime Web 版本是 `onnxruntime-web@1.27.0`。

`onnxruntime-web` 需要找到 WASM 文件。创建模型前可以这样配置：

```ts
import { initializeOnnxRuntimeWeb } from 'yolo-onnx-web';

initializeOnnxRuntimeWeb({
  wasmPaths: '/examples/browser/ort-wasm/',
});
```

也可以直接传给 `Yolo.create()` 或 `Sam3.create()`。

## 快速开始

```ts
import { Yolo } from 'yolo-onnx-web';

const yolo = await Yolo.create({
  model: '/models/yolo26s.onnx',
  wasmPaths: '/ort-wasm/',
  executionProviders: ['webgpu', 'wasm'],
});

console.log(yolo.onnxModel.modelVersion); // 例如 V26、RTDETR、RFDETR

const image = document.querySelector('img')!;
const detections = await yolo.RunObjectDetection(image, 0.2, 0.7);

const canvas = document.querySelector('canvas')!;
yolo.drawObjectDetections(image, detections, canvas);
```

## 模型来源

`model` 支持以下类型：

```ts
type YoloModelSource = string | ArrayBufferLike | Uint8Array;
```

示例：

```ts
await Yolo.create({ model: '/models/yolov8n.onnx' });
await Yolo.create({ model: new Uint8Array(await file.arrayBuffer()) });
```

## 支持模型表

支持关系来自 ONNX 元数据：

- `task`：`classify`、`detect`、`obb`、`segment`、`pose`
- `description`：用于识别模型版本
- `names`：标签映射

| 模型版本 | 分类 | 目标检测 | 旋转框检测 | 分割 | 姿态估计 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5u (`V5U`) | 否 | 是 | 否 | 否 | 否 | 复用 YOLOv8 风格检测输出 |
| YOLOv8 (`V8`) | 是 | 是 | 是 | 是 | 是 | 主要 YOLOv8 解析器 |
| YOLOv8E (`V8E`) | 否 | 否 | 否 | 是 | 否 | 仅分割 |
| YOLOv9 (`V9`) | 否 | 是 | 否 | 否 | 否 | 复用 YOLOv8 风格检测输出 |
| YOLOv10 (`V10`) | 否 | 是 | 否 | 否 | 否 | 独立 YOLOv10 检测解析器 |
| YOLO11 (`V11`) | 是 | 是 | 是 | 是 | 是 | 复用 YOLOv8 风格输出解析 |
| YOLO11E (`V11E`) | 否 | 否 | 否 | 是 | 否 | 仅分割 |
| YOLOv12 (`V12`) | 是 | 是 | 是 | 是 | 是 | 复用 YOLOv8 风格输出解析 |
| YOLO26 (`V26`) | 是 | 是 | 是 | 是 | 是 | 独立 YOLO26 解析器 |
| RT-DETR (`RTDETR`) | 否 | 是 | 否 | 否 | 否 | 独立 RT-DETR 检测解析器 |
| RF-DETR (`RFDETR`) | 否 | 是 | 否 | 是 | 否 | 独立 RF-DETR 解析器，支持检测和分割 |
| YOLO World V2 (`WORLDV2`) | 否 | 是 | 否 | 否 | 否 | 仅目标检测 |

模型加载完成后，可以读取当前 ONNX 模型的识别信息：

```ts
console.log(yolo.onnxModel.modelType);    // ObjectDetection、Segmentation 等
console.log(yolo.onnxModel.modelVersion); // V8、V26、RTDETR、RFDETR 等
console.log(yolo.onnxModel.modelDataType);
```

## 推理 API

### 分类

```ts
const results = await yolo.RunClassification(image, 5);
yolo.drawClassifications(image, results, canvas);
```

### 目标检测

```ts
const results = await yolo.RunObjectDetection(image, 0.2, 0.7);
yolo.drawObjectDetections(image, results, canvas);
```

### 旋转框检测

```ts
const results = await yolo.RunObbDetection(image, 0.2, 0.7);
yolo.drawObbDetections(image, results, canvas);
```

### 实例分割

```ts
const results = await yolo.RunSegmentation(image, 0.2, 0.65, 0.7);
yolo.drawSegmentations(image, results, canvas, {
  drawSegmentationPixelMask: true,
  pixelMaskOpacity: 128,
  drawContour: false,
});
```

### 姿态估计

```ts
const results = await yolo.RunPoseEstimation(image, 0.2, 0.7);
yolo.drawPoseEstimations(image, results, canvas, {
  poseConfidence: 0.25,
});
```

## 根据模型类型自动调用

```ts
async function runByModelType(yolo: Yolo, source: CanvasImageSource) {
  switch (yolo.onnxModel.modelType) {
    case 'Classification':
      return yolo.RunClassification(source);
    case 'ObjectDetection':
      return yolo.RunObjectDetection(source);
    case 'ObbDetection':
      return yolo.RunObbDetection(source);
    case 'Segmentation':
      return yolo.RunSegmentation(source);
    case 'PoseEstimation':
      return yolo.RunPoseEstimation(source);
  }
}
```

## 摄像头示例

```ts
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: { ideal: 640 }, height: { ideal: 640 } },
  audio: false,
});

const video = document.querySelector('video')!;
video.srcObject = stream;
await video.play();

const canvas = document.querySelector('canvas')!;

async function loop() {
  const results = await yolo.RunObjectDetection(video);
  yolo.drawObjectDetections(video, results, canvas, { drawSource: false });
  requestAnimationFrame(loop);
}

loop();
```

## 绘制选项

通用检测绘制选项：

```ts
yolo.drawObjectDetections(image, detections, canvas, {
  drawSource: true,
  drawLabel: true,
  drawConfidenceScore: true,
  drawLabelBackground: true,
  lineWidth: 2,
  font: '16px Arial',
  fontColor: '#f8fafc',
  boundingBoxHexColors: ['#22c55e', '#3b82f6'],
  boundingBoxOpacity: 255,
});
```

分割和姿态估计提供额外选项：

```ts
yolo.drawSegmentations(image, segmentations, canvas, {
  drawSegmentationPixelMask: true,
  pixelMaskOpacity: 128,
  drawContour: true,
  contourThickness: 2,
});

yolo.drawPoseEstimations(image, poses, canvas, {
  poseConfidence: 0.25,
  defaultPoseColor: '#22c55e',
  keyPointRadius: 4,
});
```

`DrawTool` 也可以作为独立绘制工具使用。适合推理和绘制分离、或者需要渲染缓存检测结果的场景：

```ts
import { DrawTool, Yolo } from 'yolo-onnx-web';

const yolo = await Yolo.create({
  model: '/models/rf-detr-seg.onnx',
  wasmPaths: '/ort-wasm/',
  executionProviders: ['webgpu', 'wasm'],
  modelVersion: 'RFDETR',
  modelType: 'Segmentation',
});

const image = document.querySelector('img')!;
const canvas = document.querySelector('canvas')!;
const segmentations = await yolo.RunSegmentation(image, 0.35, 0.5, 0.7);

segmentations.forEach(segmentation => {
  segmentation.segmentationEdgePoints = DrawTool.extractSegmentationEdgePoints(segmentation);
});

DrawTool.drawSegmentationEdgePoints(image, segmentations, canvas, {
  drawSource: true,
  drawBoundingBoxes: true,
  drawLabel: true,
  drawSegmentationPixelMask: true,
  fillSegmentationEdgePoints: true,
  resultOpacity: 0.7,
});
```

## SAM 3

`Sam3` 与 `Yolo` 是独立 API。SAM 3.1 multiplex 使用四张 ONNX 图，会缓存图像嵌入，并按提示逐步交互。它不走 `IYoloHandler`。

### 导出 ONNX

```bash
python scripts/export_sam3_onnx.py --checkpoint sam3.1_multiplex.pt --output-dir ./sam3-onnx
python scripts/export_sam3_onnx.py --output-dir ./sam3-onnx --fold-existing --convert-fp16
```

`--fold-existing` 会去掉恒为假的 `If` 节点，否则 onnxruntime-web 会在形状推断时报错。`--convert-fp16` 会生成 `*.fp16.onnx`。浏览器请加载 fp16 视觉/文本编码器。

需要的文件：

| 文件 | 作用 |
| --- | --- |
| `vision-encoder.fp16.onnx` | 图像 → 检测 FPN + PVS 嵌入 |
| `text-encoder.fp16.onnx` | CLIP token → 语言特征 |
| `grounding-decoder.onnx` | PCS（按概念找出全部实例） |
| `prompt-decoder.onnx` | PVS（只分割当前提示的那一件） |
| `clip_bpe.json` | CLIP BPE tokenizer 表 |

图像会缩放到 1008×1008，mean/std 为 `0.5`。请优先使用 WebGPU，并设置 `numThreads: 1`。

### 快速开始

```ts
import { Sam3 } from 'yolo-onnx-web';

const sam3 = await Sam3.create({
  visionEncoder: visionBytes,
  textEncoder: textBytes,
  groundingDecoder: groundingBytes,
  promptDecoder: promptBytes,
  tokenizer: '/models/clip_bpe.json',
  wasmPaths: '/ort-wasm/',
  executionProviders: ['webgpu'],
  numThreads: 1,
  confidenceThreshold: 0.5,
  onLoadProgress: message => console.log(message),
});

const image = document.querySelector('img')!;
const canvas = document.querySelector('canvas')!;
await sam3.setImage(image);

const pcs = await sam3.setTextPrompt('person, car, dog', 'wearing a red hat');
sam3.drawSegmentationEdgePoints(image, pcs, canvas);

const pvs = await sam3.addPoint({ x: 120, y: 80 }, 1);
sam3.drawSegmentationEdgePoints(image, pvs.masks, canvas);

await sam3.dispose();
```

模型来源与 `Yolo.create()` 相同，可以是 URL、`ArrayBuffer` 或 `Uint8Array`。

### 概念分割（PCS）

类别名**只按逗号**拆分（`person, car, dog` / `猫, 狗`），不要用空格拆词。可选的详细描述会拼进查询（`person, wearing a red hat`），界面标签仍用类别名。

PCS 里的框和点是范例：模型会在全图找出同类实例，而不是只分割当前画出的那一件。

```ts
await sam3.setTextPrompt('person, car');
await sam3.addGeometricPrompt({ x: 10, y: 20, width: 80, height: 120 }, true);
await sam3.addGeometricPoint({ x: 40, y: 60 }, true);
sam3.resetPrompts();
```

### 视觉分割（PVS）

PVS 只分割当前提示的那一件。点标签：`1` 前景，`0` 背景。PVS 需要 `promptDecoder`。`iou_predictions` 是未校准回归，可能大于 1；展示时请把分数限制在 `[0, 1]`。

```ts
const byPoint = await sam3.addPoint({ x: 120, y: 80 }, 1);
const byBox = await sam3.addBox({ x: 40, y: 50, width: 200, height: 160 });
sam3.resetVisualPrompts();
```

### 绘制

PCS 后处理会按 bbox 裁剪 mask，并填写 `segmentationEdgePoints`。请用 `sam3.drawSegmentationEdgePoints()` 或独立的 `DrawTool` 绘制。

```ts
sam3.drawSegmentationEdgePoints(image, masks, canvas, {
  drawSource: true,
  drawBoundingBoxes: true,
  drawLabel: true,
  drawSegmentationPixelMask: true,
  fillSegmentationEdgePoints: true,
  resultOpacity: 0.7,
});
```

## 浏览器示例

在线 Demo：

- 入口：[https://freefer.github.io/yolo-onnx-web/](https://freefer.github.io/yolo-onnx-web/)
- YOLO：[https://freefer.github.io/yolo-onnx-web/examples/browser/](https://freefer.github.io/yolo-onnx-web/examples/browser/)
- SAM 3：[https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html](https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html)

本地启动：

```bash
npm start
```

打开：

```text
https://localhost:5173/
https://localhost:5173/examples/browser/
https://localhost:5173/examples/browser/sam3.html
```

YOLO 示例页默认使用摄像头模式。如果没有选择本地模型，也没有填写模型 URL，则默认使用：

```text
/examples/model/yolo26s.onnx
```

SAM 3 示例不会托管数 GB 的 ONNX。请选择本地导出目录，或分别选择各个模型文件，再点「加载模型」。

## 构建

```bash
npm run build
npm run build:pages
```

库构建结果输出到 `dist/`。GitHub Pages Demo 构建到 `dist-example/`（含 YOLO 与 SAM 3 页面，不含 SAM 3 权重）。

## 注意事项

- YOLO 模型需要包含 Ultralytics 兼容的 ONNX metadata。
- SAM 3 图来自 `scripts/export_sam3_onnx.py`。浏览器请加载 fp16 视觉/文本编码器；全精度编码器很容易触发 WASM `bad_alloc`。
- WebGPU、WebNN、WebGL 等后端是否可用取决于浏览器和设备。
- WebGPU 通常需要 HTTPS 或 localhost 环境。
- GitHub Pages 不托管数 GB 的 SAM 3 权重。在线 SAM 3 Demo 需要你选择本地导出目录。
