"""Run a small Marabou robustness query for Assignment 3.

The model is a tiny ReLU classifier over two normalized Iris features:
petal length and petal width.  The query checks whether any input inside an
L-infinity ball around a Setosa example can make a non-Setosa logit at least
as large as the Setosa logit.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import numpy as np

try:
    from maraboupy import Marabou
except ImportError as exc:  # pragma: no cover - exercised only without setup
    raise SystemExit(
        "maraboupy is not installed. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iris_tiny.nnet"
DATASET_PATH = ROOT / "data" / "iris.csv"
RESULTS_PATH = ROOT / "results" / "verification_result.json"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisRow(NamedTuple):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


def load_iris_dataset(path: Path) -> List[IrisRow]:
    """Load the full UCI Iris CSV file bundled in data/iris.csv."""
    rows: List[IrisRow] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for raw in csv.reader(csv_file):
            if not raw:
                continue
            if len(raw) != 5:
                raise ValueError(f"Expected 5 columns in {path}, got {raw!r}")
            rows.append(
                IrisRow(
                    sepal_length=float(raw[0]),
                    sepal_width=float(raw[1]),
                    petal_length=float(raw[2]),
                    petal_width=float(raw[3]),
                    species=raw[4].replace("Iris-", "").lower(),
                )
            )
    if not rows:
        raise ValueError(f"No Iris rows found in {path}")
    return rows


def normalized_petal_features(rows: List[IrisRow], species: str) -> np.ndarray:
    """Return normalized [petal_length, petal_width] for the first row of a species."""
    petal_lengths = np.array([row.petal_length for row in rows], dtype=float)
    petal_widths = np.array([row.petal_width for row in rows], dtype=float)
    for row in rows:
        if row.species == species:
            return np.array(
                [
                    (row.petal_length - petal_lengths.min())
                    / (petal_lengths.max() - petal_lengths.min()),
                    (row.petal_width - petal_widths.min())
                    / (petal_widths.max() - petal_widths.min()),
                ],
                dtype=float,
            )
    raise ValueError(f"Could not find species {species!r} in dataset")


def evaluate_tiny_iris_model(x: np.ndarray) -> np.ndarray:
    """Evaluate the .nnet weights directly for reporting."""
    hidden = np.maximum(
        np.array(
            [
                x[0],
                x[1],
                x[0] + x[1] - 0.55,
            ],
            dtype=float,
        ),
        0.0,
    )
    weights = np.array(
        [
            [-3.0, -3.0, 0.0],
            [2.0, 1.0, -2.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    biases = np.array([2.0, 0.2, -1.0], dtype=float)
    return weights @ hidden + biases


def _flatten_vars(vars_like: Any) -> List[int]:
    return [int(v) for v in np.asarray(vars_like).flatten()]


def _stats_total_time(stats: Any, fallback_seconds: float) -> float:
    getter = getattr(stats, "getTotalTime", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            pass
    return fallback_seconds


def _solve_network(network: Any, options: Any) -> tuple[str, Dict[int, float], Any]:
    """Support Maraboupy versions that return either 2 or 3 solve values."""
    result = network.solve(verbose=False, options=options)
    if len(result) == 3:
        exit_code, vals, stats = result
    elif len(result) == 2:
        vals, stats = result
        exit_code = "sat" if len(vals) > 0 else "unsat"
    else:
        raise RuntimeError(f"Unexpected Marabou solve result: {result!r}")
    return str(exit_code).lower(), vals, stats


def run_counterexample_query(
    target_class: int,
    rival_class: int,
    center: np.ndarray,
    epsilon: float,
    timeout: int,
) -> Dict[str, Any]:
    """Ask Marabou whether a rival class can beat the target class."""
    network = Marabou.read_nnet(str(MODEL_PATH), normalize=False)
    input_vars = _flatten_vars(network.inputVars)
    output_vars = _flatten_vars(network.outputVars)

    lower = np.maximum(center - epsilon, 0.0)
    upper = np.minimum(center + epsilon, 1.0)
    for var, lo, hi in zip(input_vars, lower, upper):
        network.setLowerBound(var, float(lo))
        network.setUpperBound(var, float(hi))

    # Counterexample condition: y_target - y_rival <= 0.
    # If this is UNSAT, no point in the input box can make that rival class tie
    # or beat the target logit.
    network.addInequality(
        [output_vars[target_class], output_vars[rival_class]],
        [1.0, -1.0],
        0.0,
    )

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)
    started = time.perf_counter()
    exit_code, vals, stats = _solve_network(network, options)
    elapsed = time.perf_counter() - started
    is_sat = exit_code == "sat"
    status = exit_code.upper()

    counterexample = None
    if is_sat:
        cex_input = [float(vals[var]) for var in input_vars]
        cex_logits = evaluate_tiny_iris_model(np.array(cex_input, dtype=float))
        counterexample = {
            "input": cex_input,
            "logits": cex_logits.tolist(),
            "predicted_class": CLASS_NAMES[int(np.argmax(cex_logits))],
        }

    return {
        "rival_class": CLASS_NAMES[rival_class],
        "query": f"exists x in L_inf ball: y_{CLASS_NAMES[rival_class]} >= y_{CLASS_NAMES[target_class]}",
        "status": status,
        "runtime_seconds": _stats_total_time(stats, elapsed),
        "input_lower_bounds": lower.tolist(),
        "input_upper_bounds": upper.tolist(),
        "counterexample": counterexample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file: {MODEL_PATH}")
    if not DATASET_PATH.exists():
        raise SystemExit(f"Missing dataset file: {DATASET_PATH}")

    dataset = load_iris_dataset(DATASET_PATH)
    setosa_sample = normalized_petal_features(dataset, "setosa")
    logits = evaluate_tiny_iris_model(setosa_sample)
    target_class = int(np.argmax(logits))
    rival_classes = [idx for idx in range(len(CLASS_NAMES)) if idx != target_class]

    queries = [
        run_counterexample_query(
            target_class=target_class,
            rival_class=rival_class,
            center=setosa_sample,
            epsilon=args.epsilon,
            timeout=args.timeout,
        )
        for rival_class in rival_classes
    ]
    verified = all(query["status"] == "UNSAT" for query in queries)

    result = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_rows": len(dataset),
        "features": "normalized petal length and petal width",
        "center_input": setosa_sample.tolist(),
        "epsilon": args.epsilon,
        "center_logits": logits.tolist(),
        "center_prediction": CLASS_NAMES[target_class],
        "property": (
            "All inputs within the L-infinity ball around the Setosa sample "
            "keep the Setosa logit strictly above both non-Setosa logits."
        ),
        "verified": verified,
        "queries": queries,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print()
    print("VERIFIED" if verified else "COUNTEREXAMPLE FOUND")


if __name__ == "__main__":
    main()
