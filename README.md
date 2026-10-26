# Crossed support-recovery code

Install Python dependencies with `python -m pip install -r requirements.txt`.
Run `python reproduce_all.py` to reproduce all experiments and figures. This is a substantial computation. The wrapper downloads the public lexdec data with a SHA-256 check and sets BLAS thread counts to one.

Individual entry points:

- `run_irregular.py`: incomplete-array simulations and analytic bounds.
- `legacy/run_experiments.py`: balanced simulations and the adaptive-weight example.
- `run_revision.py`: paired REML comparisons, variance-box sensitivity and real-data illustration.
- `verify_revision.py`: algebra and likelihood checks.
- `make_explanatory_figures.py`: explanatory vector diagrams.
- `fetch_data.py`: download and verify the public languageR lexdec data.

For individual numerical scripts, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Run `python fetch_data.py` before `python run_revision.py --real-only`. Seeds are fixed in the source. Generated outputs and downloaded data stay local.
