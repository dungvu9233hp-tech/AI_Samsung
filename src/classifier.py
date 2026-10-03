import numpy as np
import onnxruntime as ort


class ONNXClassifier:
    """CNN 64x64 xám -> xác suất lớp 1 (eye: closed, yawn: yawn)."""
    def __init__(self, path, threads=2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def probs(self, crops):
        x = np.stack(crops).astype(np.float32)[:, None] / 255.0
        logits = self.s.run(None, {self.inp: x})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e[:, 1] / e.sum(1)).astype(np.float64)
