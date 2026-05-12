# Problem 1: Marabou Resources Directory

I reviewed the Marabou repository at
`https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources`.
The directory is organized around benchmark networks, properties, and scripts
that reproduce or demonstrate Marabou experiments.

## Model Types

- `resources/nnet`: `.nnet` fully connected ReLU networks. Subdirectories
  include ACAS Xu, CollisionAvoidance, MNIST, and TwinStreams benchmarks.
- `resources/onnx`: ONNX networks. The folder contains small fully connected
  examples, ACAS Xu and CIFAR-10 subfolders, MNIST models, convolutional and
  max-pooling examples, traffic-sign models, and layer/operator tests.
- `resources/keras`: Keras `.h5` MNIST CNN examples.
- `resources/bnn_queries`: parsed and original binary neural network query
  examples.
- Other resource folders include `mps`, `target`, and `img`, plus notebooks
  and helper scripts.

## Datasets and Input Specifications

- The benchmark families include ACAS Xu, CollisionAvoidance, MNIST,
  TwinStreams, CIFAR-10, traffic-sign, and small synthetic examples.
- `resources/properties` stores text property files. It has ACAS properties
  `acas_property_1.txt` through `acas_property_4.txt`, a built-in property
  file for CollisionAvoidance/TwinStreams, and MNIST property files.
- The property files describe input bounds and output inequalities. This is
  useful for robustness or reachability queries.

## Example Verification Queries

- Command-line example from the README:
  `Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet resources/properties/acas_property_3.txt`.
- Python API examples in `maraboupy/examples` demonstrate loading `.nnet`,
  `.onnx`, and TensorFlow models, setting input/output bounds, and solving.
- ONNX examples show fully connected and convolutional models, including local
  perturbation bounds over image inputs.
- The resources folder also includes `runMarabou.py` and a
  `SplitAndConquerGuide.ipynb` notebook for larger queries.

## Takeaway for Problem 2

Because the resources directory already contains many ACAS Xu, MNIST, CIFAR-10,
traffic-sign, and small ONNX examples, I used a separate tiny Iris classifier in
`.nnet` format for Problem 2. This keeps the model outside the provided
resources and small enough for Marabou to solve quickly.

