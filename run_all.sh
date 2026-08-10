#!/usr/bin/env bash
# Build and execute every notebook in order.
#   bash run_all.sh                # full run (~90 min on a 4-core CPU)
#   TLLAB_QUICK=1 bash run_all.sh  # ~10x smaller smoke run, separate artifacts
set -euo pipefail
cd "$(dirname "$0")"

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if [ ! -d .venv ]; then
  "$UV" venv .venv
fi
"$UV" pip install --quiet --python .venv/bin/python -r requirements.txt
if [ ! -d .venv/share/jupyter/kernels/python3 ]; then
  .venv/bin/python -m ipykernel install --sys-prefix --name python3
fi

total0=$SECONDS
for src in notebooks/src/[0-9][0-9]_*.py; do
  name=$(basename "$src" .py)
  echo "=== $name ==="
  t0=$SECONDS
  .venv/bin/jupytext --to notebook --execute --output "notebooks/$name.ipynb" "$src"
  echo "=== $name finished in $((SECONDS - t0))s ==="
done
echo "ALL NOTEBOOKS DONE in $((SECONDS - total0))s"
