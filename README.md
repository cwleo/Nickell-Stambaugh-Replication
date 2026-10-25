# Replication: Nickell Meets Stambaugh

**Nickell Meets Stambaugh: A Tale of Two Biases in Panel Predictive Regressions**  
Chengwang Liao, Ziwei Mei, and Zhentao Shi

Article: [https://doi.org/10.1016/j.jeconom.2026.106347](https://doi.org/10.1016/j.jeconom.2026.106347)

This repository contains the code for the simulations (Figures 2–4) and empirical application (Figure 5 and Tables 1–2) in the main text.

## Files

- `code/simulation/`: simulation estimators and figure scripts.
- `code/empirical/`: empirical estimation, tests, tables, and figures.
- `data/raw/`: empirical data from Wind Information.
- `reference/`: saved simulation results for Figures 2–4.
- `output/`: generated tables, figures, and numerical results.
- `run_all.py`: runs the replication.

## Setup

The code was tested with Python 3.12.3 and R 4.4.0. Run the following commands from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript setup.R
export R_LIBS_USER="$PWD/.Rlib"
```

The empirical scripts read `16PE-DY.csv` and `16PS-PB.csv` from `data/raw/`. See [data information](data/README.md).

## Run the replication

```sh
python run_all.py --fresh-simulation --workers 4
```

This command reproduces the empirical results and reruns the simulations using the original random seeds. The simulations take approximately seven minutes with four workers on the tested machine.

Tables and figures are saved in `output/empirical/` and `output/simulation/historical/`. Tables are exported as LaTeX files and figures as PNG and PDF files.

To recalculate the empirical results and generate the simulation figures from saved results:

```sh
python run_all.py
```

The simulation figures from this command are saved in `output/simulation/archive/`.

To run each part separately:

```sh
# Empirical application
python code/empirical/run.py
python code/empirical/export_tables.py

# Simulations using the original random seeds
python code/simulation/historical.py --workers 4
```
