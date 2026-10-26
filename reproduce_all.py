"""Run all numerical experiments and regenerate the manuscript figures."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
from fetch_data import fetch

def main():
    root = Path(__file__).resolve().parent
    fetch()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for script in ("run_irregular.py", "legacy/run_experiments.py", "run_revision.py",
                   "verify_revision.py", "check_clarifications.py", "make_explanatory_figures.py"):
        subprocess.run([sys.executable, str(root / script)], cwd=root, env=env, check=True)
    (root / "results_legacy").mkdir(exist_ok=True)
    for name in ("geometry", "mixed", "counterexample"):
        shutil.copyfile(root / "legacy/results" / (name + ".pdf"),
                        root / "results_legacy" / (name + ".pdf"))

if __name__ == "__main__":
    main()
