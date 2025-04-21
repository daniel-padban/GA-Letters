import onnx

model = onnx.load('demo/demo_models/Run-Demo-12000-S100.onnx')
input_names = [inp.name for inp in model.graph.input]
print("Model Input Names:", input_names)
print([output.name for output in model.graph.output])  # Get the output names
