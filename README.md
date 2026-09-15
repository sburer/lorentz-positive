# Reproducing the computations in the Lorentz paper

This directory is the paper-facing version of the computational code for
the manuscript by Kurt Anstreicher and Sam Burer on bilinear optimization
over Lorentz cones (`paper/Lorentz.tex` in the paper repository). It is
meant to answer two questions cleanly:

- Where did each number in the paper come from?
- How do I regenerate the tables and examples?

If you only want to check that the manuscript numbers match the committed
data, the quick rebuild below takes a couple of minutes. Rerunning the solver
experiments is a different matter: some panels take hours, and the full set
takes longer. The commands for those full runs are listed by problem family.

The file that ties everything together is `paper_experiments.toml`. For each
`tab:*` label and named example, it records the raw CSV, the script that
created it, and the LaTeX fragment generated from it. When a number looks
mysterious, start there.

## What you need

- Python 3.11 or newer with `numpy` and `scipy`
  (`pip install -r requirements.txt`).
- MOSEK 11 with the Fusion API, and Gurobi 13 via `gurobipy`. Both need a
  license. All the conic relaxations run on MOSEK; Gurobi is only used as the
  global-optimization comparison. Each solver finds its license the usual
  way (`~/mosek/mosek.lic` or `MOSEKLM_LICENSE_FILE`; `~/gurobi.lic` or a
  WLS file or `GRB_LICENSE_FILE`), and nothing here needs configuring beyond
  that.
- Optionally, [Tectonic](https://tectonic-typesetting.github.io), if you
  want the standalone PDFs of the table fragments. Without it the driver
  simply skips that step.
- The companion `ballconstraints` repository, which we vendor as a git
  submodule at `external/ballconstraints`. It provides the 212 TTRS instances
  used in Section 4.2, the beta relaxation used in the Section 4.2
  counterexample, and the conic standard-form builder that the
  noxious-facility models are built on. After cloning this repository, run

  ```bash
  git submodule update --init
  ```

  If you already have a checkout of
  <https://github.com/sburer/ballconstraints> somewhere else, point
  `BALLCONSTRAINTS_DIR` at it instead. The pinned commit is recorded under
  `[external]` in the TOML file.

All commands in this file are meant to be run from this directory, and
`python` means whatever interpreter satisfies the list above. Nothing depends
on the current working directory or on `PYTHONPATH`; every script locates
its data relative to its own file. One caveat on
timing: the wall-clock numbers in the paper (and the estimates below) come
from the machine described under `[environment]` in the TOML. Bounds and gaps
should reproduce anywhere to solver tolerance; times will not.

## Quick check: rebuild the tables from committed data

```bash
python reproduce.py             # tables, example, PDFs, checks
python reproduce.py --skip-pdf  # same, without Tectonic
python reproduce.py --check-only
```

This rebuilds every table fragment in `outputs/tables/` from the raw CSVs in
`outputs/raw/`. It also re-solves the named counterexample, rewrites its log
in `outputs/examples/`, and runs the consistency checks.

The main check is `check_paper_tables.py`. It reads each `tabular` in
`Lorentz.tex` and compares its printed numbers against the regenerated
fragment. The manuscript tables are pasted and styled by hand, so the check
does not require identical LaTeX. It only asks the important question: did
any number change?

The other checks cover the surrounding bookkeeping. They verify that the
manuscript labels match the TOML inventory, that each fragment contains the
labels it should, that methods in a raw CSV were run on the same random
instances, and that the named examples in the text are reproducible from
their source scripts.

The manuscript itself is not part of this directory. The checks look for it
at `../paper/Lorentz.tex` (the `manuscript` entry in the TOML), which is
where it sits when this directory is a subfolder of the paper repository. If
the file is not there, the three manuscript-facing checks are skipped rather
than failed, and `reproduce.py` says so; pass
`--manuscript /path/to/Lorentz.tex` to run them against a copy elsewhere.
The data-only checks always run.

`--tables-only`, `--examples-only`, and `--panel {bilinear,maxdist,ttrs,large,noxious}`
narrow the run. Nothing here ever edits `Lorentz.tex`. When a check
fails, either the manuscript or the data needs a human to fix it, and the
TOML has a `[[known_manuscript_issues]]` / `[[known_data_issues]]` section
where we note anything currently outstanding along with how to fix it.

## Directory map

| Path | What it is |
|------|------------|
| `reproduce.py` | The driver described above. |
| `paper_experiments.toml` | Manuscript-to-file inventory, plus environment, tolerances, and known issues. |
| `paper_code/` | The model implementations the runners share (listed at the end). |
| `run_*_experiments.py` | The solver panels; they write `outputs/raw/*.csv`. |
| `make_*_tables.py`, `make_current_comp_note.py` | Raw CSV to summary CSV to LaTeX fragment. |
| `check_*.py` | Consistency and correctness checks. The manuscript-level ones are run by `reproduce.py`; the model-level ones are mentioned under each family. |
| `verify_ttrs_sep_beta_counterexample.py` | The named Section 4.2 example. |
| `outputs/` | Committed raw data and everything regenerated from it; see `outputs/README.md`. |
| `external/ballconstraints` | The submodule. |

The goal is for this directory to be necessary and sufficient for the paper:
source-of-truth files, generated artifacts worth committing, and checks that
make the connection auditable. Exploratory studies, superseded panels, and
the older Burer–Dong noxious-facility formulation are kept out of it (they
live in the `code/` directory of the paper repository).

## Shared conventions

Every SDP method reports two quantities: its certified lower bound and the
objective value of a feasible point recovered from that method's own first
column. The relative gap in the tables is computed from those two numbers and
nothing else. Gurobi is kept separate: its bracket is its own incumbent and
global bound. Throughout the paper-code, "tight" means relative gap at most
`1e-5`; the tolerances are collected under `[global_tolerances]` in the TOML.

The lazy methods add violated coordinate skew-skew equalities (Lazy SEP) or
scalar Kronecker cuts (Lazy KRON) with a violation tolerance of `1e-7` and at
most `--lazy-cuts-per-iter` cuts per round.

Gurobi always runs last, because its time limit on each instance is
`max(5, 3 * t_max)` seconds where `t_max` is the slowest conic method on that
instance. In practice this means Gurobi accounts for most of the solver time
in every family, so if you only care about the conic results the panels are
much cheaper than the totals below.

The runners write rows as they go. With `--resume`, they append only missing
`(instance, method)` rows. When resuming, use the same generator flags as the
original run; otherwise the same key may refer to a different random
instance. `check_raw_provenance.py` catches that problem, but it is better not
to create it.

Smoke tests (`--quick`, `--limit`) write to separate `*_smoke.csv` files, so
you can try things without touching the committed panels.

Notation follows the paper: `x in R^m` is the source variable, `y in R^n`
the target, and the SEP lift is `Z = [[1, x'], [y, V]] in SEP(n+1, m+1)`.

## Rerunning the solver panels

Each family below lists the tables it feeds, a small smoke test, and the
commands that produced the committed CSVs. The commands are shown in the
order they were run.

### Bilinear optimization over two balls (Section 3.1; `tab:bilinear_times`, `tab:bilinear_diagnostics`)

This is problem (7): `min c'x + d'y + y'Rx` over two unit balls, with
`x in R^m` and `y in R^n`. The comparison uses Shor, Full SEP, Lazy SEP, Dual
LOP/TRS, and Gurobi. The dual method is the scalar bisection algorithm whose
separation step is solved by the exact TRS oracle.

The panel has ten instances at each square size `2x2, 4x4, 6x6, 8x8, 10x10,
15x15, 20x20`, and ten more at each rectangular size `4x8, 8x12, 10x20,
15x20, 15x25` (always written `n x m`). The instances use the
`correlated-top` generator: `R` is standard normal, the linear terms are
biased toward the leading singular vectors of `R`, and then `(c, d, R)` is
scaled by `max(||c||, ||d||)`. The scale factor is recorded in the CSV as
`data_scale`.

For each size, seeds are scanned from zero and the first ten whose Shor
relative gap exceeds `0.01` are kept. Those selected seeds are committed in
`outputs/raw/bilinear_selected_seeds.csv`. The full panel took about 11 hours
of solver time: roughly 8 hours for Gurobi and 2.5 hours for Full SEP.

```bash
# smoke test (n=m=2, one seed)
python run_bilinear_experiments.py --quick --verbose

# 1. select the panel (writes outputs/raw/bilinear_selected_seeds.csv)
python run_bilinear_experiments.py \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --cases-per-size 10 --select-only --verbose

# 2. conic methods, then Gurobi with the matched time limit
python run_bilinear_experiments.py \
  --selected-panel-input outputs/raw/bilinear_selected_seeds.csv \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --methods shor,full,lazy,dual --verbose
python run_bilinear_experiments.py \
  --selected-panel-input outputs/raw/bilinear_selected_seeds.csv \
  --generator correlated-top --generator-alpha 3 --generator-sigma 0.1 \
  --prefilter-relative-gap 0.01 --methods gurobi --resume --verbose

# 3. tables
python make_bilinear_tables.py
```

If you touch the Dual LOP/TRS code, run `check_dual_lop.py`; it exercises the
rank-one repair step on degenerate instances that the random panel never
happens to produce.

### Maximum distance between two ellipsoids (Section 4.1; `tab:maxdist_times`, `tab:maxdist_diagnostics`, `tab:maxdist_larger`)

This is problem (12) specialized to the maximum distance between two
ellipsoids:
`max ||(Ax + a) - (By + b)||^2` over two unit balls, with `n = m`.
The methods are Shor, KRON, Lazy KRON, Full SEP, Lazy SEP, and Gurobi. The
conic methods are relaxations here, so each method reports its own recovered
upper bound and relative gap.

Instances come from the default `ellipsoids` generator
(`--condition 5 --center-scale 1`): random rotations, singular values
log-uniform on `[1, 5]`, and shifted centers. Since Shor is already tight on
many candidates, the panel uses `--prefilter`. A candidate is discarded if
the Shor solution is certified rank one, either directly by its eigenvalue
ratio or after re-solving over the Shor optimal face with a random objective.

The first ten survivors per size are kept. The main panel uses
`n = 2, 4, 6, 8, 10, 15` and is recorded in
`outputs/raw/maxdist_selected_seeds.csv`; it took about 6.5 hours, including
4.7 hours for Gurobi and 1.8 for KRON. The larger add-on panel for
`tab:maxdist_larger` uses `n = 20, 25, 30` with only Lazy KRON and Lazy SEP;
that run takes minutes.

```bash
# smoke test
python run_maxdist_experiments.py --quick --verbose

# main panel: select, run the conic methods, then Gurobi, then tables
python run_maxdist_experiments.py \
  --sizes 2,4,6,8,10,15 --cases-per-size 10 --prefilter --select-only --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_selected_seeds.csv \
  --methods shor,kron,lazy-kron,full,lazy --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_selected_seeds.csv \
  --methods gurobi --resume --verbose
python make_maxdist_tables.py

# larger instances (tab:maxdist_larger)
python run_maxdist_experiments.py \
  --sizes 20,25,30 --cases-per-size 10 --prefilter --select-only \
  --selected-panel-output outputs/raw/maxdist_large_selected_seeds.csv --verbose
python run_maxdist_experiments.py \
  --selected-panel-input outputs/raw/maxdist_large_selected_seeds.csv \
  --output outputs/raw/maxdist_large_raw.csv \
  --methods lazy-kron,lazy --rank-validation-only --resume --verbose
python make_large_instance_tables.py
```

If memory is tight on the largest Full KRON models, `--mosek-threads N`
limits MOSEK's parallelism without changing the formulation or tolerances.

The model-level check for this family is `check_quad_models.py`. It verifies
that the Section 4.1 Full SEP model reduces to the Section 3 exact model when
`Q = P = 0`, that the expected bound ordering holds, and that Lazy SEP agrees
with Full SEP. Keep this check green if you change `paper_code/quad_models.py`.
The table-side checks, `check_maxdist_tables.py` and
`check_large_instance_tables.py`, guard the tightness and gap calculations.

### The 212 TTRS instances from Burer and Anstreicher (2013) (Section 4.2; `tab:TTRS_results`, `tab:TTRS_diagnostics`)

These are the centered diagonal TTRS instances
`min x'Qx + c'x` subject to `||x|| <= 1` and `||diag(a) x|| <= 1`. The 212
instances come from the 2013 paper and were exactly the cases not solved by
SOC-RLT: 38 with `n = 5`, 70 with `n = 10`, and 104 with `n = 20`. They are
read from
`external/ballconstraints/data/soctrust/Section_5_3_unsolved_instances_saved/`
and scaled into the form above by `paper_code/ttrs_instances.py`. The paper
reports Full SEP, Lazy SEP, and Gurobi. (The committed CSV also contains Lazy
KRON rows, which the tables do not use.)

This is the expensive family. The full run took about 76 hours: 43 hours for
Gurobi and 33 for Full SEP, almost all of it from the `n = 20` cases. Lazy
SEP takes minutes.

```bash
# smoke test
python run_ttrs_experiments.py \
  --limit 2 --methods full,lazy --output /tmp/ttrs_smoke.csv --verbose

# paper panel
python run_ttrs_experiments.py --methods full,lazy --verbose
python run_ttrs_experiments.py --methods gurobi --resume --verbose
python make_ttrs_tables.py
```

Because the Gurobi step is long, `--max-hours X` is useful. The runner writes
each finished row immediately, stops cleanly when the time budget is used up,
and caps the current per-instance limit by whatever budget remains.

### Larger generated TTRS instances (Section 4.2; `tab:TTRS_larger`)

These are generated centered diagonal TTRS instances with `a_i = exp(u_i)`,
`u_i` uniform on `[-1, 1]`, and standard-normal `Q` and `c`. Shor is only a
prefilter here: if it certifies a candidate as rank one, that candidate is
discarded. The panel keeps ten instances at each of `n = 20, 25, 30` and
solves them with Lazy KRON and Lazy SEP only. It runs in minutes.

```bash
python run_ttrs_generated_experiments.py \
  --n 20,25,30 --cases-per-size 10 --select-only --verbose
python run_ttrs_generated_experiments.py \
  --n 20,25,30 --cases-per-size 10 \
  --selected-panel-input outputs/raw/ttrs_generated_large_selected_seeds.csv \
  --output outputs/raw/ttrs_generated_large_raw.csv \
  --methods lazy-kron,lazy --lazy-tol 1e-7 --lazy-cuts-per-iter 100 --resume --verbose
python make_large_instance_tables.py
```

### The shifted TTRS counterexample (Section 4.2; `eq:ttrs_counterexample_data`)

A single `n = 2` instance is used as the counterexample:
`a = (4/3, 13/12)`, `b = (11/20, -1/20)`,
`Q = -(1/16) [[24, 1], [1, 13]]`, and `c = (-1, 0)`. On this instance,
Shor, Full SEP, and the beta relaxation from Burer (2025) are all inexact.

The script `verify_ttrs_sep_beta_counterexample.py` solves Shor and Full SEP
with the TTRS models here, solves the shifted beta relaxation using the
`ballconstraints` SDP builder, and certifies the true optimum by enumerating
stationary points and boundary intersections. Adding `--with-gurobi` also
confirms the optimum globally. `reproduce.py` saves the script output as
`outputs/examples/ttrs_sep_beta_counterexample_verification.txt`, and
`check_paper_examples.py` compares those values with the manuscript. A short
standalone write-up of the example is in
`outputs/examples/ttrs_sep_beta_counterexample_note.tex`.

```bash
python verify_ttrs_sep_beta_counterexample.py --with-gurobi
```

### Noxious facility location (Section 4.3; `tab:regular`, `tab:random`, `tab:targeted`)

This is the noxious-facility problem
`max theta` subject to `||x - p_i|| >= theta` for all `i` and
`x in conv{p_i}`. The code uses the unit-sphere formulation from the paper;
the derivation is in the docstring of `paper_code/noxious_models.py`.

The methods are Shor, RLT, KRON, Full SEP, and Lazy SEP. Each relaxation's
`theta_bound` is an upper bound on the distance. The exact value is computed
by enumerating the hull/Voronoi arrangement (`true_value_arrangement`), and
the tables report `theta_bound - exact`.

There are three instance families: regular `m`-gons, random points in the
unit disk normalized by their minimum enclosing disk, and the fixed four-point
instance in `eq:targeted`. Everything here runs in minutes.

```bash
# model checks
python check_noxious_models.py --solve

# tab:regular (m = 3..16), plus the m = 20, 24, 32 values quoted in the text
python run_noxious_experiments.py \
  --generator regular --sizes 3,4,5,6,7,8,10,12 --methods all \
  --output outputs/raw/noxious_regular_m3_m12.csv
python run_noxious_experiments.py \
  --generator regular --sizes 16 --methods all \
  --output outputs/raw/noxious_regular_m16.csv
python run_noxious_experiments.py \
  --generator regular --sizes 20,24,32 --methods shor,rlt,full \
  --output outputs/raw/noxious_regular_large.csv

# tab:random (m = 4, 6, 8, 15, 20 in one file; m = 10 in another)
python run_noxious_experiments.py \
  --generator random-disk --sizes 4,6,8,15,20 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --methods shor,rlt,kron,full \
  --output outputs/raw/noxious_random_panel.csv
python run_noxious_experiments.py \
  --generator random-disk --sizes 10 --seeds 0,1,2,3,4,5,6,7,8,9 --methods all \
  --output outputs/raw/noxious_random10.csv

# tab:targeted
python run_noxious_experiments.py \
  --generator targeted-410 --methods all \
  --output outputs/raw/noxious_targeted_410.csv

python make_noxious_tables.py
```

## The model code in `paper_code/`

| Module | What it does |
|--------|--------------|
| `instances.py`, `shor.py`, `full_sep.py`, `lazy_sep.py`, `dual_lop.py`, `gurobi_qp.py` | Section 3.1: bilinear instances and the five methods. |
| `quad_instances.py`, `quad_models.py`, `quad_gurobi.py` | Section 4.1: quadratic-over-balls instances (the max-distance generators) and the Shor / KRON / Lazy KRON / Full SEP / Lazy SEP family built by `solve_quad`. |
| `ttrs_instances.py`, `ttrs_models.py`, `ttrs_gurobi.py` | Section 4.2: loading and scaling the TTRS instances, the affine-diagonal SEP and KRON lifts, Gurobi. |
| `noxious_instances.py`, `noxious_models.py` | Section 4.3: geometry, generators, the exact planar reference, and the relaxations. |
| `sep_utils.py` | The Hildebrand-II SEP building blocks shared by everything: the `W` basis, skew-skew equations, coordinate and SVD cut separation. |
| `metrics.py` | Relative gaps and the tightness tolerance. |
| `external.py` | Finds the `ballconstraints` checkout. |
