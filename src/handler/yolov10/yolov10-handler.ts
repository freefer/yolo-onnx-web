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
import { unsupportedTask } from '../common';

export class Yolov10Handler implements IYoloHandler {
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
    if (this._yolo.onnxModel.modelType !== 'ObjectDetection') {
      unsupportedTask('ObjectDetection');
    }

    void iou; // YOLOv10 exported detection models are already post-processed by the model.

    const input = this.preprocessImage(img, roi);
    const result = await this._yolo.run({
      [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
    });
    const data = Object.values(result)[0].data as Float32Array;
    const detections: ObjectDetection[] = [];

    for (let i = 0; i + 5 < data.length; i += 6) {
      const score = data[i + 4];

      if (score < confidence) {
        continue;
      }

      const label = this._yolo.onnxModel.labels[Math.trunc(data[i + 5])];

      if (!label) {
        continue;
      }

      detections.push(new ObjectDetection({
        label,
        confidence: score,
        boundingBox: this._yolo.scaleBoundingBox(data[i], data[i + 1], data[i + 2], data[i + 3], input),
      }));
    }

    return detections;
  }

  RunObbDetection(img: YoloImageSource, confidence: number, iou: number, roi: Rect | null = null): Promise<OBBDetection[]> {
    void img;
    void confidence;
    void iou;
    void roi;
    unsupportedTask('ObbDetection');
  }

  RunSegmentation(
    img: YoloImageSource,
    confidence: number,
    pixelConfidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<Segmentation[]> {
    void img;
    void confidence;
    void pixelConfidence;
    void iou;
    void roi;
    unsupportedTask('Segmentation');
  }

  RunPoseEstimation(
    img: YoloImageSource,
    confidence: number,
    iou: number,
    roi: Rect | null = null,
  ): Promise<PoseEstimation[]> {
    void img;
    void confidence;
    void iou;
    void roi;
    unsupportedTask('PoseEstimation');
  }

  RunClassification(img: YoloImageSource, classes: number): Promise<Classification[]> {
    void img;
    void classes;
    unsupportedTask('Classification');
  }
}
