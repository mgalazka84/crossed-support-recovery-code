# Crossed support recovery: code

Python 3.12 scripts for synthetic crossed-model simulations.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_irregular.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python legacy/run_experiments.py
```

The scripts generate their synthetic inputs and write outputs locally to
`results/` and `legacy/results/`. Generated files are excluded from Git.
