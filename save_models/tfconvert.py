import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('save_models/onnx_models/2024-09-30 18_13_15.758843.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('save_models/tf_models/2024-09-30 18_13_15.758843.pb')

