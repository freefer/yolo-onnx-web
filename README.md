# yolo-onnx-web

Browser-side YOLO and SAM 3 inference powered by `onnxruntime-web`.

This package loads Ultralytics-style ONNX models in the browser, parses model metadata, dispatches to the correct YOLO output handler, and provides drawing helpers for detection, classification, segmentation, pose estimation, and oriented bounding boxes. It also ships `Sam3` for SAM 3.1 multiplex concept segmentation (PCS) and interactive visual segmentation (PVS).

Repository: [https://github.com/freefer/yolo-onnx-web](https://github.com/freefer/yolo-onnx-web)

Online demos: [YOLO](https://freefer.github.io/yolo-onnx-web/examples/browser/) · [SAM 3](https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html)

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
- Reuses RF-DETR WebGPU preprocess textures and buffers across frames; `yolo.dispose()` releases them.
- Adds `Sam3` for SAM 3.1 multiplex: text concept segmentation (PCS) and interactive visual segmentation (PVS).
- Exports `DrawTool` so drawing helpers can be used independently from a `Yolo` instance.
- Splits segmentation drawing: YOLO / RF-DETR use `drawSegmentationEdgePoints`; SAM 3 uses `sam3.drawSegmentations()` (`DrawTool.drawSam3Segmentations`).

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

You can also pass the same options to `Yolo.create()` or `Sam3.create()`.

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

For YOLO / RF-DETR results that you persist and redraw later, extract polygons first, then use edge-point drawing (works even after `bitPackedPixelMask` is dropped):

```ts
results.forEach(segmentation => {
  segmentation.segmentationEdgePoints = yolo.extractSegmentationEdgePoints(segmentation);
});

yolo.drawSegmentationEdgePoints(image, results, canvas, {
  drawBoundingBoxes: true,
  drawLabel: true,
  drawSegmentationPixelMask: true,
  fillSegmentationEdgePoints: true,
  resultOpacity: 0.7,
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

Segmentation drawing is split by model family:

| API | Use for | Fill source |
| --- | --- | --- |
| `yolo.drawSegmentations()` / `DrawTool.drawSegmentations()` | Live YOLO packed masks | `bitPackedPixelMask` |
| `yolo.drawSegmentationEdgePoints()` / `DrawTool.drawSegmentationEdgePoints()` | YOLO / RF-DETR, including cached JSON polygons | packed mask when present, otherwise `segmentationEdgePoints` |
| `sam3.drawSegmentations()` / `DrawTool.drawSam3Segmentations()` | SAM 3 PCS / PVS | packed mask plus contours |

Do not use `DrawTool.drawSam3Segmentations` for YOLO / RF-DETR replay. `Sam3.drawSegmentationEdgePoints` was removed; call `sam3.drawSegmentations()` instead.

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

## SAM 3

`Sam3` is a separate API from `Yolo`. SAM 3.1 multiplex uses four ONNX graphs, keeps image embeddings in memory, and applies prompts iteratively. It is not dispatched through `IYoloHandler`.

### Export ONNX

```bash
python scripts/export_sam3_onnx.py --checkpoint sam3.1_multiplex.pt --output-dir ./sam3-onnx
python scripts/export_sam3_onnx.py --output-dir ./sam3-onnx --fold-existing --convert-fp16
```

`--fold-existing` removes constant-false `If` nodes that otherwise fail onnxruntime-web shape inference. `--convert-fp16` writes `*.fp16.onnx` sidecars. In the browser, load the fp16 vision and text encoders.

Required files:

| File | Role |
| --- | --- |
| `vision-encoder.fp16.onnx` | Image → detector FPN + PVS embeddings |
| `text-encoder.fp16.onnx` | CLIP tokens → language features |
| `grounding-decoder.onnx` | PCS (find all instances of a concept) |
| `prompt-decoder.onnx` | PVS (segment the prompted object) |
| `clip_bpe.json` | CLIP BPE tokenizer tables |

Images are resized to 1008×1008 with mean/std `0.5`. Prefer WebGPU and `numThreads: 1`.

### Quick Start

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
sam3.drawSegmentations(image, pcs, canvas);

const pvs = await sam3.addPoint({ x: 120, y: 80 }, 1);
sam3.drawSegmentations(image, pvs.masks, canvas);

await sam3.dispose();
```

Model sources accept a URL, `ArrayBuffer`, or `Uint8Array`, the same as `Yolo.create()`.

### Concept segmentation (PCS)

Class names are split on commas only (`person, car, dog` / `猫, 狗`). Spaces do not split classes. The optional description is appended to each class when querying (`person, wearing a red hat`); the displayed label stays the class name.

Boxes and points in PCS are exemplars: the model finds all matching instances in the image, not only the drawn object.

```ts
await sam3.setTextPrompt('person, car');
await sam3.addGeometricPrompt({ x: 10, y: 20, width: 80, height: 120 }, true);
await sam3.addGeometricPoint({ x: 40, y: 60 }, true);
sam3.resetPrompts();
```

### Visual segmentation (PVS)

PVS segments only the prompted object. Points use `1` (foreground) or `0` (background). `promptDecoder` is required. The IoU head is uncalibrated and may exceed 1; clamp display scores to `[0, 1]`.

```ts
const byPoint = await sam3.addPoint({ x: 120, y: 80 }, 1);
const byBox = await sam3.addBox({ x: 40, y: 50, width: 200, height: 160 });
sam3.resetVisualPrompts();
```

### Drawing

PCS postprocessing packs masks by bbox and fills `segmentationEdgePoints`. Draw with `sam3.drawSegmentations()`.

```ts
sam3.drawSegmentations(image, masks, canvas, {
  drawSource: true,
  drawBoundingBoxes: true,
  drawLabel: true,
  drawSegmentationPixelMask: true,
  fillSegmentationEdgePoints: true,
  resultOpacity: 0.7,
});
```

## Browser Demo

Online demos:

- Hub: [https://freefer.github.io/yolo-onnx-web/](https://freefer.github.io/yolo-onnx-web/)
- YOLO: [https://freefer.github.io/yolo-onnx-web/examples/browser/](https://freefer.github.io/yolo-onnx-web/examples/browser/)
- SAM 3: [https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html](https://freefer.github.io/yolo-onnx-web/examples/browser/sam3.html)

Run locally:

```bash
npm start
```

Open:

```text
https://localhost:5173/
https://localhost:5173/examples/browser/
https://localhost:5173/examples/browser/sam3.html
```

The YOLO demo defaults to camera mode and uses `/examples/model/yolo26s.onnx` if no model file or URL is provided. The SAM 3 demo does not bundle the multi-GB ONNX files; choose a local export directory or pick each file, then click Load.

## Build

```bash
npm run build
npm run build:pages
```

The package is built with `tsup` into `dist/`. The GitHub Pages demo is built into `dist-example/` (YOLO + SAM 3). SAM 3 weights are not bundled.

## Notes

- YOLO models should include Ultralytics-compatible ONNX metadata.
- SAM 3 graphs come from `scripts/export_sam3_onnx.py`. In the browser, load fp16 vision/text encoders; full-precision encoders often hit WASM `bad_alloc`.
- Call `yolo.dispose()` when replacing a model or tearing down the page so RF-DETR WebGPU preprocess textures/buffers are released.
- Browser support depends on the selected execution provider and the user's device/browser.
- For WebGPU, use a browser with WebGPU enabled and HTTPS or localhost.
- GitHub Pages does not host the multi-GB SAM 3 weights. The online SAM 3 demo asks you to select a local export directory.
