#!/usr/bin/env python
import os # Needed to read datafiles
import numpy as np # Data is stored as an ndarray
import onnx # ONNX models are used
import ngraph as ng # backend to run
from ngraph_onnx.onnx_importer.importer import import_onnx_model

# First get the model(s) that will be ran
model_dir = 'models'
for model in os.listdir(model_dir):
  print("Using model", model)
  modelpath = os.path.join(model_dir, model)
  onnx_protobuf = onnx.load(modelpath) # Load model in ONNX
  ng_models = import_onnx_model(onnx_protobuf) # Import to ngraph-onnx
  ng_model = ng_models[0]
  runtime = ng.runtime(backend_name='CPU') # Set the backend for CPU
  # Extract the model information with the nGraph backend for the inference model
  compute_model = runtime.computation(ng_model['output'], *ng_model['inputs'])

  # Run inference for each data sample
  data_dir = 'data'
  for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    sample = np.load(filepath, encoding='bytes')
    input_data = np.array(sample['inputs'])
    # Make sure the input data is in the proper shape
    input_shape = ng_model['inputs'][0].get_shape()
    input_data = input_data.reshape(input_shape)
    # Check with the proper output
    expected_output = np.array(sample['outputs'])
    # Compute the inference model result, and reshape
    result = np.array(compute_model(input_data))
    result = result.reshape(expected_output.shape)
    # Check that the expected and actual output are the same
    np.testing.assert_allclose(result, expected_output, rtol=0.001)
print("All Tests Passed")
