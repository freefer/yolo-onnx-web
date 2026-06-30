# yolo-onnx-web

Browser-side YOLO inference powered by `onnxruntime-web`.

This package loads Ultralytics-style ONNX models in the browser, parses model metadata, dispatches to the correct YOLO output handler, and provides drawing helpers for detection, classification, segmentation, pose estimation, and oriented bounding boxes.

Repository: [https://github.com/freefer/yolo-onnx-web](https://github.com/freefer/yolo-onnx-web)

Chinese documentation: [README.zh-CN.md](https://github.com/freefer/yolo-onnx-web/blob/main/README.zh-CN.md)

## Features

- Runs ONNX models in the browser with `onnxruntime-web`.
- Supports WebGPU, WASM, WebNN, WebGL, and CPU execution providers where available.
- Parses ONNX custom metadata: `task`, `description`, and `names`.
- Supports image and camera/video sources.
- Provides high-level APIs for:
  - Classification
  - Object detection
  - Oriented bounding box detection
  - Segmentation
  - Pose estimation
- Includes canvas drawing utilities for all supported task types.
- Supports RF-DETR object detection and segmentation models.
- Exports `DrawTool` so drawing helpers can be used independently from a `Yolo` instance.

## Installation

```bash
npm install yolo-onnx-web
```

If you are working from this repository:

```bash
npm install
npm run build
npm start
```

The browser example runs at the fixed Vite port configured in `vite.config.ts`.

## Browser Runtime Setup

This package currently depends on `onnxruntime-web@1.27.0`.

`onnxruntime-web` needs access to its WASM files. Configure the path before creating a model:

```ts
import { initializeOnnxRuntimeWeb } from 'yolo-onnx-web';

initializeOnnxRuntimeWeb({
  wasmPaths: '/examples/browser/ort-wasm/',
});
```

You can also pass the same options to `Yolo.create()`.

## Quick Start

```ts
import { Yolo } from 'yolo-onnx-web';

const yolo = await Yolo.create({
  model: '/models/yolo26s.onnx',
  wasmPaths: '/ort-wasm/',
  executionProviders: ['webgpu', 'wasm'],
});

console.log(yolo.onnxModel.modelVersion); // e.g. V26, RTDETR, RFDETR

const image = document.querySelector('img')!;
const detections = await yolo.RunObjectDetection(image, 0.2, 0.7);

const canvas = document.querySelector('canvas')!;
yolo.drawObjectDetections(image, detections, canvas);
```

## Model Sources

`model` can be one of:

```ts
type YoloModelSource = string | ArrayBufferLike | Uint8Array;
```

Examples:

```ts
await Yolo.create({ model: '/models/yolov8n.onnx' });
await Yolo.create({ model: new Uint8Array(await file.arrayBuffer()) });
```

## Supported Models

Support is selected from the ONNX metadata:

- `task`: `classify`, `detect`, `obb`, `segment`, or `pose`
- `description`: used to infer model version
- `names`: label map

| Model version | Classification | Object detection | OBB detection | Segmentation | Pose estimation | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv5u (`V5U`) | No | Yes | No | No | No | Uses YOLOv8-style detection output |
| YOLOv8 (`V8`) | Yes | Yes | Yes | Yes | Yes | Main YOLOv8 handler |
| YOLOv8E (`V8E`) | No | No | No | Yes | No | Segmentation only |
| YOLOv9 (`V9`) | No | Yes | No | No | No | Uses YOLOv8-style detection output |
| YOLOv10 (`V10`) | No | Yes | No | No | No | Dedicated YOLOv10 detection handler |
| YOLO11 (`V11`) | Yes | Yes | Yes | Yes | Yes | Uses YOLOv8-style output handlers |
| YOLO11E (`V11E`) | No | No | No | Yes | No | Segmentation only |
| YOLOv12 (`V12`) | Yes | Yes | Yes | Yes | Yes | Uses YOLOv8-style output handlers |
| YOLO26 (`V26`) | Yes | Yes | Yes | Yes | Yes | Dedicated YOLO26 handlers |
| RT-DETR (`RTDETR`) | No | Yes | No | No | No | Dedicated RT-DETR detection handler |
| RF-DETR (`RFDETR`) | No | Yes | No | Yes | No | Dedicated RF-DETR handler; supports detection and segmentation |
| YOLO World V2 (`WORLDV2`) | No | Yes | No | No | No | Object detection only |

You can inspect the current ONNX model metadata after loading:

```ts
console.log(yolo.onnxModel.modelType);    // ObjectDetection, Segmentation, ...
console.log(yolo.onnxModel.modelVersion); // V8, V26, RTDETR, RFDETR, ...
console.log(yolo.onnxModel.modelDataType);
```

## Inference APIs

### Classification

```ts
const results = await yolo.RunClassification(image, 5);
yolo.drawClassifications(image, results, canvas);
```

### Object Detection

```ts
const results = await yolo.RunObjectDetection(image, 0.2, 0.7);
yolo.drawObjectDetections(image, results, canvas);
```

### Oriented Bounding Box Detection

```ts
const results = await yolo.RunObbDetection(image, 0.2, 0.7);
yolo.drawObbDetections(image, results, canvas);
```

### Segmentation

```ts
const results = await yolo.RunSegmentation(image, 0.2, 0.65, 0.7);
yolo.drawSegmentations(image, results, canvas, {
  drawSegmentationPixelMask: true,
  pixelMaskOpacity: 128,
  drawContour: false,
});
```

### Pose Estimation

```ts
const results = await yolo.RunPoseEstimation(image, 0.2, 0.7);
yolo.drawPoseEstimations(image, results, canvas, {
  poseConfidence: 0.25,
});
```

## Unified Dispatch Example

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

## Camera Example

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

## Drawing Options

Common detection drawing options:

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

Segmentation and pose drawing expose extra options:

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

`DrawTool` is also exported as a standalone helper. This is useful when inference and drawing live in different modules, or when you want to render cached results:

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

## Browser Demo

Run:

```bash
npm start
```

Open:

```text
https://localhost:5173/examples/browser/
```

The demo defaults to camera mode and uses `/examples/model/yolo26s.onnx` if no model file or URL is provided.

## Build

```bash
npm run build
```

The package is built with `tsup` into `dist/`.

## Notes

- Models should include Ultralytics-compatible ONNX metadata.
- Browser support depends on the selected execution provider and the user's device/browser.
- For WebGPU, use a browser with WebGPU enabled and HTTPS or localhost.
