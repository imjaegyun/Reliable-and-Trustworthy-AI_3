# Marabou Assignment 3

This repository contains a small Marabou verification experiment for Reliable
and Trustworthy Artificial Intelligence Assignment 3.

## Files

- `test.py`: loads the model, formulates the robustness query, runs Marabou,
  and writes `results/verification_result.json`.
- `models/iris_tiny.nnet`: a small fully connected ReLU model in Marabou's
  `.nnet` format.
- `data/iris_reference.csv`: the Iris examples used to explain the selected
  input domain.
- `resources_exploration.md`: summary of the Marabou `resources` directory.
- `report.pdf`: short assignment report.

## Setup

The recommended Marabou installation method is pip:

```bash
python -m pip install -r requirements.txt
```

Marabou's README says this installs both the `Marabou` executable and the
Python bindings. The current Python interface supports Python 3.8 through 3.11.

## Run

```bash
python test.py
```

Optional:

```bash
python test.py --epsilon 0.04 --timeout 30
```

The script runs two Marabou queries. Each query asks whether one non-Setosa
class can tie or beat the Setosa logit inside the input perturbation box. If
both queries return `UNSAT`, the local robustness property is verified for this
epsilon.

