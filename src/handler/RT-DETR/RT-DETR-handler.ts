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
import { clamp, unsupportedTask } from "../common";

export class RT_DETRHandler implements IYoloHandler{
    private _yolo: Yolo;
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

        void iou;

        const input = this.preprocessImage(img, roi);
        const result = await this._yolo.run({
            [input.inputName]: this._yolo.tensor('float32', input.tensorData, input.inputShape),
        });
        const output = Object.values(result)[0];
        const data = output.data as Float32Array;
        const outputShape = Object.values(this._yolo.onnxModel.outputShapes)[0];

        if (!outputShape || outputShape.length < 3) {
            throw new Error(`Unsupported RT-DETR output shape: ${JSON.stringify(outputShape)}`);
        }

        const predictions = outputShape[1];
        const attributes = outputShape[2];
        const totalLabels = this._yolo.onnxModel.labels.length;
        const inputWidth = input.inputShape[3];
        const inputHeight = input.inputShape[2];
        const detections: ObjectDetection[] = [];

        for (let prediction = 0; prediction < predictions; prediction += 1) {
            const offset = prediction * attributes;
            const x = data[offset];
            const y = data[offset + 1];
            const w = data[offset + 2];
            const h = data[offset + 3];

            let labelIndex = -1;
            let score = 0;

            for (let c = 0; c < totalLabels; c += 1) {
                const classScore = data[offset + 4 + c];

                if (classScore > score) {
                    score = classScore;
                    labelIndex = c;
                }
            }

            if (labelIndex < 0 || score < confidence) {
                continue;
            }

            const label = this._yolo.onnxModel.labels[labelIndex];

            if (!label) {
                continue;
            }

            const halfWidth = w * inputWidth * 0.5 * input.gain;
            const halfHeight = h * inputHeight * 0.5 * input.gain;
            const centerX = (x * inputWidth - input.xPad) * input.gain;
            const centerY = (y * inputHeight - input.yPad) * input.gain;
            const roiLeft = input.roi?.left ?? 0;
            const roiTop = input.roi?.top ?? 0;

            detections.push(new ObjectDetection({
                label,
                confidence: score,
                boundingBox: {
                    left: roiLeft + this.clamp(Math.trunc(centerX - halfWidth), 0, input.sourceWidth - 1),
                    top: roiTop + this.clamp(Math.trunc(centerY - halfHeight), 0, input.sourceHeight - 1),
                    right: roiLeft + this.clamp(Math.trunc(centerX + halfWidth), 0, input.sourceWidth - 1),
                    bottom: roiTop + this.clamp(Math.trunc(centerY + halfHeight), 0, input.sourceHeight - 1),
                },
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

    private clamp(value: number, min: number, max: number): number {
        return clamp(value, min, max);
    }
}