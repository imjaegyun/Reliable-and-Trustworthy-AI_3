"""Run a small Marabou robustness query for Assignment 3.

The model is a tiny ReLU classifier over two normalized Iris features:
petal length and petal width.  The query checks whether any input inside an
L-infinity ball around a Setosa example can make a non-Setosa logit at least
as large as the Setosa logit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

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
RESULTS_PATH = ROOT / "results" / "verification_result.json"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
SETOSA_SAMPLE = np.array([0.0677966102, 0.0416666667], dtype=float)


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
    vals, stats = network.solve(verbose=False, options=options)
    elapsed = time.perf_counter() - started
    is_sat = len(vals) > 0

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
        "status": "SAT" if is_sat else "UNSAT",
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

    logits = evaluate_tiny_iris_model(SETOSA_SAMPLE)
    target_class = int(np.argmax(logits))
    rival_classes = [idx for idx in range(len(CLASS_NAMES)) if idx != target_class]

    queries = [
        run_counterexample_query(
            target_class=target_class,
            rival_class=rival_class,
            center=SETOSA_SAMPLE,
            epsilon=args.epsilon,
            timeout=args.timeout,
        )
        for rival_class in rival_classes
    ]
    verified = all(query["status"] == "UNSAT" for query in queries)

    result = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "dataset": "Iris, using normalized petal length and petal width",
        "center_input": SETOSA_SAMPLE.tolist(),
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

