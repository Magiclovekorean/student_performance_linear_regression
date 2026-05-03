# AGENTS.md

## Core Rules
- Use only NumPy; no high-level ML libraries (scikit-learn, etc.)
- Follow modular pipeline structure per README: preprocessing → split → model → train → evaluate

## Repo-Specific Facts
- `main.py` is the pipeline entry point (currently empty)
- `data/dataset.csv` is the raw input dataset; avoid unrecorded modifications
- `src/` modules are empty stubs mapped to pipeline steps: `preprocessing.py`, `split.py`, `model.py`, `train.py`, `evaluate.py`
- Existing `venv/` is tracked in git (contents ignored via `venv/.gitignore`); use `venv/bin/pip` for package installs
- README references `config.py` and `src/data.py` which don't exist yet; align new files with README's stated structure

## Commands
- Install dependencies: `venv/bin/pip install numpy`
- Run pipeline: `venv/bin/python main.py` (once implemented)
