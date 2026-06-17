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
} from '../../types';
import { Yolo } from '../../yolo';
import {
  DecodedObject,
  downscaleBoxToMask,
  packSegmentationMask,
  removeOverlappingBoxes,
  scalePoint,
  toDetection,
  unsupportedTask,
} from '../common';

export class Yolov8Handler implements IYoloHandler {
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

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const output = Object.values(result)[0];
    const objects = this.decodeObjectDetections(output.data as Float32Array, input, confidence, iou);

    return objects.map(item => new ObjectDetection(toDetection(item)));
  }

  async RunObbDetection(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<OBBDetection[]> {
    this.ensureTask('ObbDetection');

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const objects = this.decodeObjectDetections(Object.values(result)[0].data as Float32Array, input, confidence, iou);

    return objects.map(item => new OBBDetection({
      ...toDetection(item),
      orientationAngle: item.orientationAngle ?? 0,
    }));
  }

  async RunSegmentation(
    img: YoloImageSource,
    confidence: number,
    pixelConfidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<Segmentation[]> {
    this.ensureTask('Segmentation');

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const outputs = Object.values(result);
 
    const detectionData = outputs[0].data as Float32Array;
    const maskData = outputs[1]?.data as Float32Array | undefined;
  
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
    const segmentations: Segmentation[] = [];
    const maskWeights = new Float32Array(maskChannels);

    for (const item of objects) {
      let maskOffset = item.boundingBoxIndex + predictions * (labels + 4);

      for (let channel = 0; channel < maskChannels; channel += 1, maskOffset += predictions) {
        maskWeights[channel] = detectionData[maskOffset] ?? 0;
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
        pixelConfidence,
      );

      segmentations.push(new Segmentation({
        ...toDetection(item),
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

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const output = Object.values(result)[0];
    const data = output.data as Float32Array;
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

    return objects.map(item => {
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
          confidence: data[cIndex] ?? 0,
        });
      }

      return new PoseEstimation({
        ...toDetection(item),
        keyPoints,
      });
    });
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

  private decodeObjectDetections(
    data: Float32Array,
    input: YoloPreprocessResult,
    confidence: number,
    iou: number,
  ): DecodedObject[] {
    const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];

    if (!outputShape || outputShape.length < 3) {
      throw new Error(`Unsupported YOLOv8 output shape: ${JSON.stringify(outputShape)}`);
    }

    const predictions = outputShape[2];
    const labels = this._yolo.onnxModel.labels.length;
    const attribute2 = predictions * 2;
    const attribute3 = predictions * 3;
    const attribute4 = predictions * 4;
    const objects: DecodedObject[] = [];

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
        orientationAngle: data[i + predictions * (4 + labels)] ?? 0,
      });
    }

    return removeOverlappingBoxes(objects, iou);
  }

  private ensureTask(task: 'Classification' | 'ObjectDetection' | 'ObbDetection' | 'Segmentation' | 'PoseEstimation'): void {
    if (this._yolo.onnxModel.modelType !== task) {
      unsupportedTask(task);
    }
  }
}
