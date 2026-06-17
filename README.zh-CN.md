# yolo-onnx-web

基于 `onnxruntime-web` 的浏览器端 YOLO 推理库。

本库可以在浏览器中加载 Ultralytics 风格的 ONNX 模型，解析模型元数据，根据模型版本和任务类型自动选择输出解析器，并提供分类、检测、分割、姿态估计、旋转框等任务的 Canvas 绘制方法。

English documentation: [README.md](./README.md)

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

`onnxruntime-web` 需要找到 WASM 文件。创建模型前可以这样配置：

```ts
import { initializeOnnxRuntimeWeb } from 'yolo-onnx-web';

initializeOnnxRuntimeWeb({
  wasmPaths: '/examples/browser/ort-wasm/',
});
```

也可以直接传给 `Yolo.create()`。

## 快速开始

```ts
import { Yolo } from 'yolo-onnx-web';

const yolo = await Yolo.create({
  model: '/models/yolo26s.onnx',
  wasmPaths: '/ort-wasm/',
  executionProviders: ['webgpu', 'wasm'],
});

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
| YOLO World V2 (`WORLDV2`) | 否 | 是 | 否 | 否 | 否 | 仅目标检测 |

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

## 浏览器示例

启动：

```bash
npm start
```

打开：

```text
https://localhost:5173/examples/browser/
```

示例页默认使用摄像头模式。如果没有选择本地模型，也没有填写模型 URL，则默认使用：

```text
/examples/model/yolo26s.onnx
```

## 构建

```bash
npm run build
```

构建结果输出到 `dist/`。

## 注意事项

- 模型需要包含 Ultralytics 兼容的 ONNX metadata。
- WebGPU、WebNN、WebGL 等后端是否可用取决于浏览器和设备。
- WebGPU 通常需要 HTTPS 或 localhost 环境。
