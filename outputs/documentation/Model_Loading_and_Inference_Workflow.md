# Model Loading and Inference Workflow

```python
import torch
import numpy as np

# 1. Load PyTorch Checkpoint
checkpoint = torch.load("outputs/models/best_model.pt", map_location="cpu")
print("Loaded model trained at epoch:", checkpoint.get("epoch", "N/A"))

# 2. Or Load Optimized TorchScript Binary
model_scripted = torch.jit.load("outputs/models/best_model_scripted.pt")
model_scripted.eval()

# 3. Or Load ONNX Runtime Session (Ultra-Lightweight C++ Engine)
import onnxruntime as ort
session = ort.InferenceSession("outputs/models/best_model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 4. Execute Offline Inference
dummy_input = np.random.randn(1, 7, 128).astype(np.float32)
pred_prob = session.run([output_name], {input_name: dummy_input})[0]
print("Calibrated Malignancy Risk:", pred_prob[0][0])
```
