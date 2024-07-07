const Yolov8 = require('./lib/yolov8');

module.exports = function (RED) {
  function yolov8Run(config) {
    RED.nodes.createNode(this, config);
    const topk = parseInt(config.topk || 1);
    const iouThreshold = parseFloat(config.iouThreshold || 0.5);
    const confidenceThreshold = parseFloat(config.confidenceThreshold || 0.25);
    const node = this;
    const yolov8 = new Yolov8(
      null,
      topk,
      iouThreshold,
      confidenceThreshold
    );
    node.on('input', async function (msg, send, done) {
      send = send || function () { node.send.apply(node, arguments) };
      done = done || function (err) { if (err) { node.error(err, msg) } };
      node.status({ fill: "blue", shape: "dot", text: "processing" });
      yolov8.detect(msg.payload).then((result) => {
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
      }).catch((err) => {
        node.status({ fill: "red", shape: "ring", text: "error" });
        done(err);
      });
    });
  }
  RED.nodes.registerType("obj detection", yolov8Run);
}
