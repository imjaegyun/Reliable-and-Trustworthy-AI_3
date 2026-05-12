#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ASSIGNMENT3_CONDA_ENV:-assignment3-marabou}"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniconda3/bin/conda" "/data/home/ijg2603/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      export PATH="$(dirname "$candidate"):$PATH"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda or add conda to PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$ENV_NAME" == "assignment3-marabou" ]]; then
    conda env create -f environment.yml
  else
    conda create -y -n "$ENV_NAME" python=3.10 pip
  fi
fi

conda activate "$ENV_NAME"
python -m pip install -r requirements.txt
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/assignment3_pycache"

mapfile -t python_files < <(find . -maxdepth 2 -name "*.py" -not -path "./.git/*" -not -path "./.venv/*" | sort)
if [[ "${#python_files[@]}" -gt 0 ]]; then
  python -m py_compile "${python_files[@]}"
fi

python test.py "$@"
