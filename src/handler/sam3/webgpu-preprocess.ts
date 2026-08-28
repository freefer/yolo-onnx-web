import type * as OrtTypes from 'onnxruntime-web';
import { ort } from '../../runtime';
import type { YoloImageSource } from '../../types';
import { getImageSize, SAM3_IMAGE_SIZE, SAM3_MEAN, SAM3_STD, renderSam3ImageToCanvas } from './preprocess';
import type { Sam3ImageTensor } from './types';

const PREPROCESS_SHADER = `
struct Params {
  size: vec4<f32>,
  scale: vec4<f32>,
  bias: vec4<f32>,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(params.size.x);
  let height = u32(params.size.y);

  if (id.x >= width || id.y >= height) {
    return;
  }

  let rgba = textureLoad(inputTexture, vec2<i32>(i32(id.x), i32(id.y)), 0);
  let pixel = id.y * width + id.x;
  let planeSize = width * height;

  outputTensor[pixel] = rgba.r * params.scale.x + params.bias.x;
  outputTensor[planeSize + pixel] = rgba.g * params.scale.y + params.bias.y;
  outputTensor[planeSize * 2u + pixel] = rgba.b * params.scale.z + params.bias.z;
}
`;

async function getOrtWebGpuDevice(): Promise<any> {
  const webgpu = (ort.env as { webgpu?: { device?: any } }).webgpu;
  const device = await webgpu?.device;

  if (!device) {
    throw new Error('ONNX Runtime WebGPU device is not initialized.');
  }

  return device;
}

/**
 * session.run with gpu-buffer outputs returns after recording commands, not after the GPU finishes.
 * Queue fence only — extra copy/mapAsync on FPN outputs was extra GPU work and did not make encode faster.
 */
export async function waitForWebGpuOutputs(_result?: OrtTypes.InferenceSession.OnnxValueMapType): Promise<void> {
  const device = await (ort.env as { webgpu?: { device?: Promise<any> | any } }).webgpu?.device;
  if (typeof device?.queue?.onSubmittedWorkDone === 'function') {
    await device.queue.onSubmittedWorkDone();
  }
}

/**
 * Canvas → WebGPU NCHW float32 tensor (mean/std 0.5).
 * Avoids getImageData + JS loops + CPU upload of the 1008×1008 image.
 */
export class Sam3WebGpuPreprocessor {
  private device: any = null;
  private pipeline: any = null;
  private texture: any = null;
  private outputBuffer: any = null;
  private paramsBuffer: any = null;
  private size = 0;

  async process(image: YoloImageSource, imageSize = SAM3_IMAGE_SIZE): Promise<Sam3ImageTensor> {
    const { source, sourceWidth, sourceHeight, close } = await rasterizeSam3Square(image, imageSize);
    const device = await this.ensureResources(imageSize);

    try {
      device.queue.copyExternalImageToTexture({ source }, { texture: this.texture }, { width: imageSize, height: imageSize });
    } finally {
      close?.();
    }

    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: this.outputBuffer } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(imageSize / 16), Math.ceil(imageSize / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);

    return {
      data: new Float32Array(0),
      width: imageSize,
      height: imageSize,
      sourceWidth,
      sourceHeight,
      tensor: ort.Tensor.fromGpuBuffer(this.outputBuffer, {
        dataType: 'float32',
        dims: [1, 3, imageSize, imageSize],
      }),
    };
  }

  dispose(): void {
    this.texture?.destroy();
    this.outputBuffer?.destroy();
    this.paramsBuffer?.destroy();
    this.texture = null;
    this.outputBuffer = null;
    this.paramsBuffer = null;
    this.pipeline = null;
    this.device = null;
    this.size = 0;
  }

  private async ensureResources(imageSize: number): Promise<any> {
    const device = await getOrtWebGpuDevice();
    const usage = (globalThis as any).GPUBufferUsage;
    const textureUsage = (globalThis as any).GPUTextureUsage;

    if (!usage || !textureUsage) {
      throw new Error('WebGPU buffer usage flags are not available.');
    }

    if (this.device !== device || this.size !== imageSize || !this.pipeline || !this.texture || !this.outputBuffer || !this.paramsBuffer) {
      this.dispose();
      this.device = device;
      this.size = imageSize;
      this.texture = device.createTexture({
        size: [imageSize, imageSize, 1],
        format: 'rgba8unorm',
        usage: textureUsage.TEXTURE_BINDING | textureUsage.COPY_DST | textureUsage.RENDER_ATTACHMENT,
      });
      this.outputBuffer = device.createBuffer({
        size: 3 * imageSize * imageSize * Float32Array.BYTES_PER_ELEMENT,
        usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST,
      });
      this.paramsBuffer = device.createBuffer({
        size: 48,
        usage: usage.UNIFORM | usage.COPY_DST,
      });
      this.pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({ code: PREPROCESS_SHADER }),
          entryPoint: 'main',
        },
      });
      const scale = 1 / SAM3_STD;
      const bias = -SAM3_MEAN / SAM3_STD;
      device.queue.writeBuffer(
        this.paramsBuffer,
        0,
        new Float32Array([imageSize, imageSize, 0, 0, scale, scale, scale, 0, bias, bias, bias, 0]),
      );
    }

    return device;
  }
}

async function rasterizeSam3Square(
  image: YoloImageSource,
  imageSize: number,
): Promise<{ source: ImageBitmap | HTMLCanvasElement; sourceWidth: number; sourceHeight: number; close?: () => void }> {
  const { width: sourceWidth, height: sourceHeight } = getImageSize(image);

  try {
    const bitmap = await createImageBitmap(image as ImageBitmapSource, {
      resizeWidth: imageSize,
      resizeHeight: imageSize,
      resizeQuality: 'high',
    });
    return { source: bitmap, sourceWidth, sourceHeight, close: () => bitmap.close() };
  } catch {
    const { canvas } = renderSam3ImageToCanvas(image, imageSize, false);
    return { source: canvas, sourceWidth, sourceHeight };
  }
}
