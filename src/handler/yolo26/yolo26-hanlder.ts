import {
  Classification,
  IYoloHandler,
  OBBDetection,
  ObjectDetection,
  PoseEstimation,
  Rect,
  Segmentation,
  YoloImageSource,
  YoloPreprocessResult,
} from "../../types";
import { Yolo } from "../../yolo";
import {
  DecodedObject,
  downscaleBoxToMask,
  packSegmentationMask,
  scalePoint,
  toDetection,
  unsupportedTask,
} from "../common";

export class Yolo26Handler implements IYoloHandler {
  private readonly _yolo: Yolo;

  constructor(yolo: Yolo) {
    this._yolo = yolo;
  }

  preprocessImage(img: YoloImageSource, roi: Rect | null = null): YoloPreprocessResult {
    return this._yolo.preprocessImage(img, roi);
  }

  async RunObjectDetection(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<ObjectDetection[]> {
    this.ensureTask('ObjectDetection');
    void iou; // YOLO26 exported detection models are already post-processed by the model.

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const output = Object.values(result)[0];
    const detections = this.decodeFlatDetections(output.data as Float32Array, input, confidence);

    return detections.map(item => new ObjectDetection(toDetection(item)));
  }

  async RunObbDetection(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<OBBDetection[]> {
    this.ensureTask('ObbDetection');
    void iou;

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const data = Object.values(result)[0].data as Float32Array;
    const stride = this.getStride();
    const detections: OBBDetection[] = [];

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
          input,
        ),
        orientationAngle: data[i + 6],
      }));
    }

    return detections;
  }

  async RunSegmentation(
    img: YoloImageSource,
    confidence: number,
    pixelConfidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<Segmentation[]> {
    this.ensureTask('Segmentation');
    void iou;

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const outputs = Object.values(result);
    const data = outputs[0].data as Float32Array;
    const maskData = outputs[1]?.data as Float32Array | undefined;
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
    const segmentations: Segmentation[] = [];

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
        input,
      );
      const maskWeights: number[] = [];

      for (let channel = 0; channel < maskChannels; channel += 1) {
        maskWeights.push(data[i + 6 + channel] ?? 0);
      }

      const bitPackedPixelMask = packSegmentationMask(
        maskData,
        maskWeights,
        maskWidth,
        maskHeight,
        downscaleBoxToMask(boundingBoxUnscaled, maskWidth, maskHeight, inputWidth, inputHeight),
        boundingBox.right - boundingBox.left,
        boundingBox.bottom - boundingBox.top,
        pixelConfidence,
      );

      segmentations.push(new Segmentation({
        label,
        confidence: score,
        boundingBox,
        bitPackedPixelMask,
      }));
    }

    return segmentations;
  }

  async RunPoseEstimation(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<PoseEstimation[]> {
    this.ensureTask('PoseEstimation');
    void iou;

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const data = Object.values(result)[0].data as Float32Array;
    const stride = this.getStride();
    const dimensions = 6;
    const keypointDimensions = 3;
    const totalKeypoints = Math.floor((stride - dimensions) / keypointDimensions);
    const detections: PoseEstimation[] = [];

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
          confidence: data[offset + 2] ?? 0,
        });
      }

      detections.push(new PoseEstimation({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], input),
        keyPoints,
      }));
    }

    return detections;
  }

  async RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]> {
    this.ensureTask('Classification');

    const input = this.preprocessImage(img, null);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const output = Object.values(result)[0];
    const data = Array.from(output.data as Float32Array);

    return data
      .map((confidence, index) => ({
        label: this._yolo.onnxModel.labels[index]?.name ?? String(index),
        confidence,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, classes)
      .map(item => new Classification(item.label, item.confidence));
  }

  private decodeFlatDetections(data: Float32Array, input: YoloPreprocessResult, confidence: number): DecodedObject[] {
    const detections: DecodedObject[] = [];

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
        boundingBoxIndex: i,
      });
    }

    return detections;
  }

  private getStride(): number {
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];

    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported YOLO26 output shape: ${JSON.stringify(outputShape)}`);
    }

    return outputShape[2];
  }

  private ensureTask(task: 'Classification' | 'ObjectDetection' | 'ObbDetection' | 'Segmentation' | 'PoseEstimation'): void {
    if (this._yolo.onnxModel.modelType !== task) {
      unsupportedTask(task);
    }
  }
}