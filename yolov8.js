const Yolov8 = require('./lib/yolov8');
const path = require('path');

module.exports = function (RED) {
  function yolov8Run(config) {
    RED.nodes.createNode(this, config);
    const topk = parseInt(config.topk || 1);
    const iouThreshold = parseFloat(config.iouThreshold || 0.5);
    const confidenceThreshold = parseFloat(config.confidenceThreshold || 0.25);
    const modelPath = config.modelPath ? path.resolve(config.modelPath) : null;
    const node = this;

    try {
      const yolov8 = new Yolov8(
        modelPath,
        topk,
        iouThreshold,
        confidenceThreshold
      );

      node.on('input', async function (msg, send, done) {
        send = send || function () { node.send.apply(node, arguments) };
        done = done || function (err) { if (err) { node.error(err, msg) } };
        node.status({ fill: "blue", shape: "dot", text: "processing" });

        try {
          const result = await yolov8.detect(msg.payload);
          const unique = [...new Set(result.map((r) => r.className))];

          if (unique.length > 0) {
            node.status({ fill: "green", shape: "dot", text: `${unique.length} detected` });
          } else {
            node.status({ fill: "yellow", shape: "dot", text: "no detection" });
          }

          msg.detected = unique;
          msg.annotations = result;
          send(msg);
          done();
        } catch (err) {
          node.status({ fill: "red", shape: "ring", text: "error" });
          done(err);
        }
      });
    } catch (err) {
      node.status({ fill: "red", shape: "ring", text: "error" });
      node.error(`Failed to load model: ${err.message}`);
    }
  }
  RED.nodes.registerType("obj detection", yolov8Run);
}
