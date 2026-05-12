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

## Setup with Conda

The repository is set up to run with a conda environment:

```bash
conda env create -f environment.yml
conda activate assignment3-marabou
```

You can also let the helper script create or reuse the environment:

```bash
./run_all.sh
```

## Run

```bash
./run_all.sh
```

Optional:

```bash
./run_all.sh --epsilon 0.04 --timeout 30
```

The script runs two Marabou queries. Each query asks whether one non-Setosa
class can tie or beat the Setosa logit inside the input perturbation box. If
both queries return `UNSAT`, the local robustness property is verified for this
epsilon.
