# Companion code for *Applications of the Lorentz positive cone in nonconvex quadratic optimization*

This repository accompanies the paper *Applications of the Lorentz positive
cone in nonconvex quadratic optimization* by Samuel Burer and Kurt M.
Anstreicher. It contains the implementations, instance generators, committed
results, and table-building scripts used for the paper's computational study.

The code supports three levels of reproducibility:

1. Rebuild all eleven paper tables from the committed raw results.
2. Run small smoke tests that exercise the optimization models and checks.
3. Rerun the complete computational panels, including the longer MOSEK and
   Gurobi experiments.

The repository also includes a standalone verifier for the two-dimensional
TTRS counterexample in Section 4.2. That verifier independently evaluates the
Shor, SEP, and beta relaxations and computes the true optimum.

## Results covered by this repository

| Paper location | Problem family | Methods represented here |
|---|---|---|
| Section 3.1, Tables 1--2 | Bilinear optimization over two ellipsoids | Shor, Full SEP, Lazy SEP, dual LOP/TRS, Gurobi |
| Section 4.1, Tables 3--5 | Maximum distance between two ellipsoids | Shor, Full and Lazy KRON, Full and Lazy SEP, Gurobi |
| Section 4.2, Tables 6--8 | Two-trust-region subproblem (TTRS) | Full SEP, Lazy SEP, Lazy KRON, Gurobi |
| Section 4.2, equation (24) | Shifted two-dimensional TTRS counterexample | Shor, Full SEP, beta relaxation, exact solution |
| Section 4.3, Tables 9--11 | Noxious facility location | Shor, RLT, KRON, Full SEP, Lazy SEP |

The raw CSV files used for the paper are committed under `outputs/raw/`.
Summary CSVs and LaTeX tables are derived from those files, so the published
numbers can be checked without rerunning the expensive solver experiments.

## Quick start

Clone the repository together with its `ballconstraints` dependency:

```bash
git clone --recurse-submodules <repository-url>
cd lorentz-positive
```

If the repository was cloned without submodules, initialize the dependency
afterward:

```bash
git submodule update --init --recursive
```

Create an environment with the packages in `requirements.txt`, configure the
MOSEK and Gurobi licenses described below, and run:

```bash
python reproduce.py --skip-pdf
```

This regenerates all table fragments and summary CSVs from the committed raw
results, verifies the named examples, and runs the consistency checks. It does
not rerun the complete solver panels.

To generate standalone PDFs of the table fragments as well, install Tectonic
and omit `--skip-pdf`:

```bash
python reproduce.py
```

## Software requirements

- Python 3.11 or newer with NumPy and SciPy:

  ```bash
  python -m pip install -r requirements.txt
  ```

- MOSEK 11 with the Fusion API and a valid license. The conic formulations
  use MOSEK.
- Gurobi 13 through `gurobipy` and a valid license. Gurobi supplies the global
  optimization comparisons.
- [Tectonic](https://tectonic-typesetting.github.io) is optional and is used
  only to compile the standalone LaTeX fragments.
- The `ballconstraints` submodule under `external/ballconstraints`. It supplies
  the 212 TTRS instances, the beta-relaxation implementation used for the
  Section 4.2 counterexample, and a conic standard-form builder used by the
  noxious-facility models.

The solvers use their normal license discovery mechanisms. No solver-specific
paths are hard-coded in this repository. An existing checkout of
[`ballconstraints`](https://github.com/sburer/ballconstraints) can be used in
place of the submodule by setting `BALLCONSTRAINTS_DIR`.

The committed timings were obtained on the machine and software versions
recorded in `paper_experiments.toml`. Bounds and gaps should reproduce on
other machines to solver tolerance, but wall-clock times naturally vary.

All commands below are intended to be run from the repository root.

## Rebuilding and checking the committed results

The main reproduction commands are:

```bash
python reproduce.py             # rebuild tables and examples, compile PDFs, run checks
python reproduce.py --skip-pdf  # same checks without compiling PDFs
python reproduce.py --check-only
```

The driver reads `paper_experiments.toml`, which records the eleven tables,
the named examples, the data files supporting each result, the computational
environment, and the numerical tolerances. It rebuilds:

- `outputs/summary/`: aggregates by problem size;
- `outputs/tables/`: LaTeX table fragments and standalone PDFs;
- `outputs/examples/`: verification output for the Section 4.2 counterexample.

The checks verify the table dimensions and entries, raw-data provenance,
agreement between full and lazy formulations where required, consistency of
the selected instance panels, and the numerical values of the named examples.
If a copy of the paper's LaTeX source is available, it can be supplied with
`--manuscript /path/to/source.tex` to compare the regenerated numbers directly
with the paper. The data and model checks do not require the paper source.

The following options restrict the reproduction run:

```bash
python reproduce.py --tables-only
python reproduce.py --examples-only
python reproduce.py --panel bilinear
python reproduce.py --panel maxdist
python reproduce.py --panel ttrs
python reproduce.py --panel large
python reproduce.py --panel noxious
```

## Numerical conventions

For each SDP method, the reported lower bound comes from the relaxation and
the feasible upper bound comes from the point recovered from that method's own
first column. No Gurobi incumbent or point recovered by another method is used
to improve an SDP method's gap. Gurobi is reported separately using its own
incumbent and global bound.

The relative gap is

```text
(upper_bound - lower_bound) / max(1, |upper_bound|, |lower_bound|).
```

A result is classified as tight when this relative gap is at most `1e-5`.
Lazy SEP and Lazy KRON use a relative separation tolerance of `1e-7`. The
full list of numerical tolerances is under `[global_tolerances]` in
`paper_experiments.toml`.

The lazy methods use coordinate cuts:

- Lazy SEP adds violated coordinate skew-skew equations.
- Lazy KRON adds scalar inequalities obtained from negative eigenvectors of
  the affine Kronecker matrix.

Gurobi is run after the conic methods. Its time limit for an instance is
`max(5, 3*t_max)` seconds, where `t_max` is the longest conic-method time on
that instance. This gives Gurobi a problem-specific comparison budget while
keeping its results separate from the SDP bounds and recovered points.

Experiment runners write each completed row immediately. The `--resume`
option skips rows already present in the output CSV. A resumed run must use
the same generator and tolerance options as the original run; the provenance
checks detect mismatches. Smoke-test options such as `--quick` and `--limit`
write to separate files and do not modify the committed panels.

The notation in the code follows the paper: `x` is in `R^m`, `y` is in
`R^n`, and the SEP lift has the block form
`Z = [[1, x'], [y, V]] in SEP(n+1, m+1)`.

## Rerunning the experiments

The commands in this section recreate the committed raw result panels. Some
complete runs take many hours. For a quick model check, begin with the smoke
test shown for each family.

### Bilinear optimization over two ellipsoids

Section 3.1 studies

```text
minimize    c'x + d'y + y'Rx
subject to  ||x|| <= 1, ||y|| <= 1.
```

Tables 1--2 compare Shor, Full SEP, Lazy SEP, the dual LOP/TRS method, and
Gurobi. The dual method uses scalar bisection with an exact trust-region
subproblem separation oracle.

The panel contains ten retained instances at each square size
`2x2, 4x4, 6x6, 8x8, 10x10, 15x15, 20x20` and each rectangular size
`4x8, 8x12, 10x20, 15x20, 15x25`. The `correlated-top` generator biases the
linear terms toward leading singular vectors of `R`, then scales `(c,d,R)` so
that `max(||c||,||d||) = 1`. The first ten seeds at each size whose Shor
relative gap exceeds `0.01` are retained in
`outputs/raw/bilinear_selected_seeds.csv`.

```bash
# Smoke test
python run_bilinear_experiments.py --quick --verbose

# Select the panel
python run_bilinear_experiments.py \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --cases-per-size 10 --select-only --verbose

# Run the conic methods and then Gurobi
python run_bilinear_experiments.py \
  --selected-panel-input outputs/raw/bilinear_selected_seeds.csv \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --methods shor,full,lazy,dual --verbose
python run_bilinear_experiments.py \
  --selected-panel-input outputs/raw/bilinear_selected_seeds.csv \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --methods gurobi --resume --verbose

python make_bilinear_tables.py
```

The full panel required about eleven hours on the reference machine. The
targeted dual-method check is available as:

```bash
python check_dual_lop.py
```

### Maximum distance between two ellipsoids

Section 4.1 specializes the general quadratic problem to

```text
maximize    ||(Ax + a) - (By + b)||^2
subject to  ||x|| <= 1, ||y|| <= 1.
```

Tables 3--5 compare Shor, Full and Lazy KRON, Full and Lazy SEP, and Gurobi.
Each conic method reports the feasible point recovered from its own solution.

The default generator uses random rotations, singular values log-uniform on
`[1,5]`, and shifted centers. Because Shor is exact on many candidates, a
rank-one screen selects ten nontrivial instances at each dimension. Tables
3--4 use `n = 2,4,6,8,10,15`; Table 5 uses `n = 20,25,30` and runs only Lazy
KRON and Lazy SEP.

```bash
# Smoke test
python run_maxdist_experiments.py --quick --verbose

# Tables 3--4: select and solve the main panel
python run_maxdist_experiments.py \
  --sizes 2,4,6,8,10,15 --cases-per-size 10 --prefilter --select-only --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_selected_seeds.csv \
  --methods shor,kron,lazy-kron,full,lazy --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_selected_seeds.csv \
  --methods gurobi --resume --verbose
python make_maxdist_tables.py

# Table 5: larger instances
python run_maxdist_experiments.py \
  --sizes 20,25,30 --cases-per-size 10 --prefilter --select-only \
  --selected-panel-output outputs/raw/maxdist_large_selected_seeds.csv --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_large_selected_seeds.csv \
  --output outputs/raw/maxdist_large_raw.csv \
  --methods lazy-kron,lazy --rank-validation-only --resume --verbose
python make_large_instance_tables.py
```

The main panel required about 6.5 hours on the reference machine. The model
and table checks for this family are:

```bash
python check_quad_models.py
python check_maxdist_tables.py
python check_large_instance_tables.py
```

### Two-trust-region subproblem

Tables 6--7 use the 212 centered diagonal TTRS instances from Burer and
Anstreicher (2013) that were not solved by SOC-RLT: 38 instances with `n=5`,
70 with `n=10`, and 104 with `n=20`. In the normalization used by the code,

```text
minimize    x'Qx + c'x
subject to  ||x|| <= 1,
            ||diag(a)x|| <= 1.
```

The tables report Full SEP, Lazy SEP, and Gurobi. The raw CSV also retains
Lazy KRON results for comparison. The complete panel required about 76 hours
on the reference machine, almost entirely from Full SEP and Gurobi at `n=20`;
Lazy SEP completes in minutes.

```bash
# Smoke test
python run_ttrs_experiments.py \
  --limit 2 --methods full,lazy --output /tmp/ttrs_smoke.csv --verbose

# Tables 6--7
python run_ttrs_experiments.py --methods full,lazy --verbose
python run_ttrs_experiments.py --methods gurobi --resume --verbose
python make_ttrs_tables.py
```

For a long Gurobi run, `--max-hours X` stops cleanly after the requested wall
time. Completed rows are preserved and a later `--resume` continues with the
remaining instances.

Table 8 uses additional generated centered diagonal TTRS instances. Shor is
used only as a rank-one prefilter, and ten retained instances at each of
`n=20,25,30` are solved with Lazy KRON and Lazy SEP:

```bash
python run_ttrs_generated_experiments.py \
  --n 20,25,30 --cases-per-size 10 --select-only --verbose
python run_ttrs_generated_experiments.py \
  --n 20,25,30 --cases-per-size 10 \
  --selected-panel-input outputs/raw/ttrs_generated_large_selected_seeds.csv \
  --output outputs/raw/ttrs_generated_large_raw.csv \
  --methods lazy-kron,lazy --lazy-tol 1e-7 \
  --lazy-cuts-per-iter 100 --resume --verbose
python make_large_instance_tables.py
```

### Shifted TTRS counterexample

The two-dimensional example in Section 4.2 uses

```text
a = (4/3, 13/12)
b = (11/20, -1/20)
Q = -(1/16) [[24, 1], [1, 13]]
c = (-1, 0).
```

For this instance, Shor, Full SEP, and the beta relaxation from Burer (2025)
are all inexact. The verifier solves all three relaxations and computes the
true optimum by enumerating stationary points and boundary intersections:

```bash
python verify_ttrs_sep_beta_counterexample.py
```

Adding `--with-gurobi` provides an independent global-optimization check:

```bash
python verify_ttrs_sep_beta_counterexample.py --with-gurobi
```

The generated verification log and standalone LaTeX note are written under
`outputs/examples/`.

### Noxious facility location

Section 4.3 considers

```text
maximize    theta
subject to  ||x - p_i|| >= theta  for every i,
            x in conv{p_i}.
```

Tables 9--11 compare Shor, RLT, KRON, Full SEP, and Lazy SEP. The relaxation
value is an upper bound on the distance. The exact planar value is computed by
enumerating the convex-hull and Voronoi arrangement, and the reported gap is
the relaxation bound minus this exact value.

The experiment families are regular polygons, random points in the unit disk
normalized by their minimum enclosing disk, and the fixed four-point example
from Section 4.3. These experiments complete in minutes.

```bash
# Model checks
python check_noxious_models.py --solve

# Table 9: regular polygons
python run_noxious_experiments.py \
  --generator regular --sizes 3,4,5,6,7,8,10,12 --methods all \
  --output outputs/raw/noxious_regular_m3_m12.csv
python run_noxious_experiments.py \
  --generator regular --sizes 16 --methods all \
  --output outputs/raw/noxious_regular_m16.csv
python run_noxious_experiments.py \
  --generator regular --sizes 20,24,32 --methods shor,rlt,full \
  --output outputs/raw/noxious_regular_large.csv

# Table 10: normalized random instances
python run_noxious_experiments.py \
  --generator random-disk --sizes 4,6,8,15,20 \
  --seeds 0,1,2,3,4,5,6,7,8,9 --methods shor,rlt,kron,full \
  --output outputs/raw/noxious_random_panel.csv
python run_noxious_experiments.py \
  --generator random-disk --sizes 10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 --methods all \
  --output outputs/raw/noxious_random10.csv

# Table 11: fixed four-point instance
python run_noxious_experiments.py \
  --generator targeted-410 --methods all \
  --output outputs/raw/noxious_targeted_410.csv

python make_noxious_tables.py
```

## Repository guide

| Path | Purpose |
|---|---|
| `reproduce.py` | Rebuilds tables and examples and runs the consistency checks. |
| `paper_experiments.toml` | Inventory of tables, examples, source data, environment information, and tolerances. |
| `paper_code/` | Shared instance generators and solver implementations. |
| `run_*_experiments.py` | Runs the computational panels and writes raw CSV rows. |
| `make_*_tables.py` | Converts raw CSVs into summaries and LaTeX table fragments. |
| `check_*.py` | Checks model equivalence, table calculations, provenance, and reproducibility. |
| `verify_ttrs_sep_beta_counterexample.py` | Reproduces the shifted TTRS counterexample from Section 4.2. |
| `outputs/raw/` | Committed solver results and selected-seed files. |
| `outputs/summary/` | Aggregated results generated from the raw CSVs. |
| `outputs/tables/` | Generated LaTeX fragments and standalone PDFs. |
| `outputs/examples/` | Counterexample verification output and standalone note. |
| `external/ballconstraints/` | Pinned external data and beta-relaxation dependency. |

The main implementation modules are:

| Module | Paper role |
|---|---|
| `instances.py`, `shor.py`, `full_sep.py`, `lazy_sep.py`, `dual_lop.py`, `gurobi_qp.py` | Bilinear instances and the methods compared in Section 3.1. |
| `quad_instances.py`, `quad_models.py`, `quad_gurobi.py` | Maximum-distance instances and the Shor, KRON, and SEP formulations in Section 4.1. |
| `ttrs_instances.py`, `ttrs_models.py`, `ttrs_gurobi.py` | Loading, scaling, and solving the TTRS instances in Section 4.2. |
| `noxious_instances.py`, `noxious_models.py` | Noxious-facility geometry, exact planar reference, and relaxations in Section 4.3. |
| `sep_utils.py` | Shared Hildebrand SEP representation, skew-skew equations, and cut separation. |
| `metrics.py` | Relative-gap and tightness calculations. |
| `external.py` | Locates the `ballconstraints` dependency. |

See `outputs/README.md` for a concise description of which output files are
primary data and which are regenerated artifacts.
