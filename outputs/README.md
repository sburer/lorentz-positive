# Paper outputs

Everything under this directory is either committed input data for the paper
tables or a file that `reproduce.py` regenerates from it.

| Directory   | Contents | Produced by |
|-------------|----------|-------------|
| `raw/`      | One CSV per experiment family with one row per (instance, method), plus the `*_selected_seeds.csv` panel definitions. **These are the committed results behind the paper**; regenerating them means rerunning the solver panels (hours). | `run_*_experiments.py` |
| `summary/`  | Per-size aggregates derived from `raw/`. | `make_*_tables.py` |
| `tables/`   | LaTeX fragments, one `\begin{table}` per manuscript table label, plus `current_comp_results_note.tex` combining all of them with prose. | `make_*_tables.py`, `make_current_comp_note.py` |
| `examples/` | The named Section 4.2 counterexample: verification log (regenerated) and a standalone note. | `verify_ttrs_sep_beta_counterexample.py` |

`paper_experiments.toml` maps each manuscript table label and named
example to its raw file, generating script, and fragment.
