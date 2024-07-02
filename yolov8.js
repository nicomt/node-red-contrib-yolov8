const Yolov8 = require('./lib/yolov8');

module.exports = function (RED) {
  function yolov8Run(config) {
    RED.nodes.createNode(this, config);
    const node = this;
    const yolov8 = new Yolov8();
    node.on('input', async function (msg, send, done) {
      send = send || function () { node.send.apply(node, arguments) };
      done = done || function (err) { if (err) { node.error(err, msg) } };
      yolov8.detect(msg.payload).then((result) => {
        const unique = [...new Set(result.map((r) => r.label))];
        msg.detected = unique;
        msg.annotations = result;
        send(msg);
        done();
      }).catch((err) => {
        done(err);
      });
    });
  }
  RED.nodes.registerType("obj detection", yolov8Run);
}
