import torch

model = torch.load('models/2024-09-30 11_11_20.042451')

torch.onnx.export(model,)

