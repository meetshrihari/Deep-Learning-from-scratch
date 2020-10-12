import onnxruntime as rt
import onnx
import sys
onnx = str(sys.argv[1])
sess = rt.InferenceSession(onnx)
print(sess)