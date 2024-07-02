// Heavily derived from Hyuto (https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py)
const sharp = require('sharp');
const { Tensor, InferenceSession } = require('onnxruntime-node');
const classes = require('./classes');

class Yolov8 {
  constructor(modelPath = null, topk = 100, iouThreshold = 0.45, confidenceThreshold = 0.25) {
    this.topk = topk;
    this.iouThreshold = iouThreshold;
    this.confidenceThreshold = confidenceThreshold;
    this.imageSize = 640;
    this.modelInputShape = [1, 3, this.imageSize, this.imageSize];
    this.modelPath = modelPath || `${__dirname}/model/yolov8n.onnx`
    this.loaded = false;
  }

  async load() {
    this.yolov8 = await InferenceSession.create(this.modelPath);
    this.nms = await InferenceSession.create(`${__dirname}/model/nms-yolov8.onnx`);

    // warmup main model
    const tensor = new Tensor(
      'float32',
      new Float32Array(this.modelInputShape.reduce((a, b) => a * b)),
      this.modelInputShape
    );
    await this.yolov8.run({ images: tensor });
  }

  async detect(imageBinary) {
    if (!this.loaded) {
      await this.load();
      this.loaded = true;
    }
    const image = await sharp(imageBinary);
    const { width, height } = await image.metadata();
    const bytes = await image
      .resize({
        width: this.imageSize,
        height: this.imageSize,
        position: 'left top',
        fit: 'contain',
        kernel: 'linear',
        background: { r: 114, g: 114, b: 114 }
      })
      .removeAlpha()
      .raw()
      .toBuffer();


    const channelOffset = this.imageSize * this.imageSize;
    const float32Array = new Float32Array(this.modelInputShape.reduce((a, b) => a * b));

    // WHC -> CHW | RGB -> BGR
    for (let i = 0; i < bytes.length; i += 3) {
      const pos = Math.floor(i / 3);
      float32Array[pos] = bytes[i + 2] / 255; // B
      float32Array[channelOffset + pos] = bytes[i + 1] / 255; // G
      float32Array[channelOffset * 2 + pos] = bytes[i] / 255; // R
    }

    const tensor = new Tensor(
      'float32',
      float32Array,
      this.modelInputShape
    );

    const config = new Tensor(
      'float32',
      new Float32Array([
        this.topk, // topk per class
        this.iouThreshold, // iou threshold
        this.confidenceThreshold, // score threshold
      ])
    );

    const { output0 } = await this.yolov8.run({ images: tensor });
    const { selected } = await this.nms.run({ detection: output0, config: config });

    const imLength = Math.max(width, height);
    const scale = imLength / this.imageSize;

    const boxes = [];
    for (let idx = 0; idx < selected.dims[1]; idx++) {
      const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
      const box = data.slice(0, 4);
      const scores = data.slice(4); // classes probability scores
      const score = Math.max(...scores); // maximum probability scores
      const classId = scores.indexOf(score); // class id of maximum probability scores

      boxes.push({
        type: 'rect',
        label: `${classes[classId]} (${(score * 100).toFixed(2)}%)`,
        classId: classId,
        className: classes[classId],
        probability: score,
        bbox: [
          (box[0] - (0.5 * box[2])) * scale,
          (box[1] - (0.5 * box[3])) * scale,
          box[2] * scale,
          box[3] * scale,
        ]
      });
    }

    return boxes;
  }

}


module.exports = Yolov8;
