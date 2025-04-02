// Heavily derived from Hyuto (https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py)
const sharp = require('sharp');
const { Tensor, InferenceSession } = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

class Yolov8 {
  constructor(modelPath = null, topk = 100, iouThreshold = 0.45, confidenceThreshold = 0.25) {
    this.topk = topk;
    this.iouThreshold = iouThreshold;
    this.confidenceThreshold = confidenceThreshold;
    this.imageSize = 640;
    this.modelInputShape = [1, 3, this.imageSize, this.imageSize];
    this.modelPath = this.resolveModelPath(modelPath);
    this.classes = this.loadClasses(this.modelPath);
    this.loaded = false;
  }

  resolveModelPath(modelPath) {
    // Default to yolov8n.onnx if no path provided
    const defaultModelPath = path.join(__dirname, 'model', 'yolov8n.onnx');

    if (!modelPath) {
      return defaultModelPath;
    }

    const stats = fs.statSync(modelPath);
    if (stats.isDirectory()) {
      // Find all .onnx files in the directory
      const files = fs.readdirSync(modelPath);
      const onnxFiles = files
        .filter(file => !path.basename(file).startsWith('nms-'))
        .filter(file => path.extname(file).toLowerCase() === '.onnx');

      if (onnxFiles.length === 0) {
        throw new Error(`No .onnx files found in directory: ${modelPath}.`);
      }
      if (onnxFiles.length > 1) {
        throw new Error([
          `Multiple .onnx files found in directory: ${modelPath}.`,
          'Specify model path explicitly or move the model file to a directory without other .onnx files.'
        ].join(' '));
      }

      return path.join(modelPath, onnxFiles[0]);
    } else {
      // It's a file, check if it's an .onnx file
      const ext = path.extname(modelPath).toLowerCase();
      if (ext !== '.onnx') {
        console.warn(`Warning: The specified model file does not have .onnx extension: ${modelPath}`);
      }
      return modelPath;
    }
  }

  loadClasses(modelPath) {
    // Default classes path
    const classesSearchPath = []; // Default classes path lowest priority

    // Get directory and filename without extension
    const modelDir = path.dirname(modelPath);
    const modelNameWithExt = path.basename(modelPath);
    const modelName = path.parse(modelNameWithExt).name;

    classesSearchPath.push(path.join(modelDir, `${modelName}.classes.txt`)); // Add {modelName}.classes.txt highest priority
    classesSearchPath.push(path.join(modelDir, 'classes.txt')); // Add classes.txt in the same directory as the model
    classesSearchPath.push(path.join(modelDir, '..', 'classes.txt')); // Add classes.txt in the parent directory

    for (const classesPath of classesSearchPath) {
      try {
        const classesText = fs.readFileSync(classesPath, 'utf-8');
        return classesText.trim().split('\n').map(line => line.trim());
      } catch (error) {
        if (error.code !== 'ENOENT') {
          throw new Error(`Error reading classes file ${classesPath}: ${error.message}`);
        }
        // Ignore errors and try the next path
      }
    }

    // If no classes file found, return default classes
    const defaultClassesPath = path.join(__dirname, 'model', 'classes.txt');
    const classesText = fs.readFileSync(defaultClassesPath, 'utf-8');
    return classesText.trim().split('\n').map(line => line.trim());
  }

  // Rest of the implementation remains the same as in the original yolov8.js
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
        label: `${this.classes[classId]} (${(score * 100).toFixed(2)}%)`,
        classId: classId,
        className: this.classes[classId],
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
