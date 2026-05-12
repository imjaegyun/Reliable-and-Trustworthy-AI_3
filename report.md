# Assignment 3 Report: Marabou Verification on a Tiny Iris Classifier

## Marabou Resource Exploration

Marabou's `resources` directory contains benchmark networks and properties in
several formats. The `nnet` folder includes fully connected ReLU benchmarks such
as ACAS Xu, CollisionAvoidance, MNIST, and TwinStreams. The `onnx` folder
contains small fully connected models, convolutional examples, MNIST/CIFAR-style
models, traffic-sign models, and operator tests. There are also Keras `.h5`
MNIST CNN files, binary neural network query examples, property files, helper
scripts, and a Split-and-Conquer notebook. The property files express input
bounds and output inequalities, while the Python examples show the same idea
through the Marabou API.

## Model, Dataset, and Query

For the external experiment, I used a tiny `.nnet` model that is not from the
Marabou resources directory. The input domain is based on two Iris dataset
features: petal length and petal width, normalized to `[0, 1]` using the usual
Iris feature ranges. The model is intentionally small: two inputs, three ReLU
hidden units, and three linear output logits for Setosa, Versicolor, and
Virginica. I chose this size because Marabou is complete but can become slow on
large networks, so a small fully connected network makes the verification query
easy to inspect and reproduce.

The checked input is the normalized Setosa example
`x = [0.0677966102, 0.0416666667]`, corresponding to petal length 1.4 cm and
petal width 0.2 cm. At this point, the model logits are approximately
`[1.6716, 0.3773, -0.8072]`, so the predicted class is Setosa. The robustness
property uses an `L_inf` ball with epsilon `0.04`. Since Marabou properties are
conjunctive, I checked two counterexample queries separately:

- Does there exist an input in the epsilon box where Versicolor logit is at
  least the Setosa logit?
- Does there exist an input in the epsilon box where Virginica logit is at
  least the Setosa logit?

For each rival class, Marabou solves the counterexample condition
`y_setosa - y_rival <= 0`. If both queries are `UNSAT`, no adversarial input
exists in this perturbation region.

## Results and Interpretation

The provided `test.py` writes the result to
`results/verification_result.json`. With epsilon `0.04`, both rival-class
queries are expected to return `UNSAT`, so the Setosa prediction is verified
locally for that perturbation box. This result is plausible because the center
point has a large margin between the Setosa logit and the two non-Setosa
logits, and the perturbation region stays near the low-petal-length,
low-petal-width area associated with Setosa.

The main strength I observed is that Marabou gives a formal answer to a precise
property rather than just sampling perturbed inputs. The main limitation is
scalability: the query must be small and piecewise-linear enough for the solver
to finish comfortably. The API is straightforward for input bounds and linear
output constraints, but disjunctive robustness properties still need to be split
into multiple queries or encoded explicitly.

## Reproducibility

The repository includes `requirements.txt`, `test.py`, the `.nnet` model,
reference data, this report source, and a README with setup and run commands.
The exact verification result can be regenerated with:

```bash
python test.py --epsilon 0.04 --timeout 30
```

