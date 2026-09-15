#!/usr/bin/env python3
"""Run generated large affine-diagonal TTRS instances.

The large paper panel uses Shor only as a cheap prefilter and then solves the
selected instances with the requested large-scale method, normally Lazy SEP.
The table generator calls a method tight only when its method-specific relative
recovered-point gap is at most ``1e-5``.  Rank-one and feasibility diagnostics
are recorded separately and do not enter that classification.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_code.metrics import relative_error
from paper_code.ttrs_instances import TtrsInstance, random_centered_diag_ttrs
from paper_code.ttrs_models import solve_ttrs_relax
from run_ttrs_experiments import FIELDNAMES, METHOD_ALIASES, result_row


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "raw" / "ttrs_generated_large_raw.csv"
DEFAULT_SELECTED = ROOT / "outputs" / "raw" / "ttrs_generated_large_selected_seeds.csv"


SELECTED_FIELDNAMES = [
    "n",
    "seed",
    "generator",
    "loga_radius",
    "screen_method",
    "screen_status",
    "screen_lower_bound",
    "screen_upper_bound",
    "screen_relative_certification_gap",
    "screen_relative_rank1_residual",
    "screen_rank1_eig_ratio",
]


def parse_dimensions(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("at least one dimension is required")
    return values


def parse_methods(text: str) -> tuple[str, ...]:
    methods = []
    for part in text.split(","):
        alias = part.strip().lower()
        if not alias:
            continue
        if alias not in METHOD_ALIASES:
            raise ValueError(f"unknown method alias {alias!r}")
        methods.append(METHOD_ALIASES[alias])
    if not methods:
        raise ValueError("at least one method is required")
    return tuple(dict.fromkeys(methods))


def generated_instance(n: int, seed: int, args: argparse.Namespace) -> TtrsInstance:
    if args.generator != "centered":
        raise ValueError(f"unsupported generator {args.generator!r}")
    return random_centered_diag_ttrs(n, seed, loga_radius=args.loga_radius)


def is_rank_one_certified(row: dict[str, object], args: argparse.Namespace) -> bool:
    cert = _float(row.get("relative_certification_gap"))
    rank = _float(row.get("relative_rank1_residual"))
    feas = _float(row.get("primal_feasibility_violation"))
    eig_ratio = _float(row.get("rank1_eig_ratio"))
    if eig_ratio is not None and eig_ratio > args.screen_eig_ratio_threshold:
        return True
    return (
        cert is not None
        and rank is not None
        and feas is not None
        and cert <= args.screen_cert_tol
        and rank <= args.screen_rank_tol
        and feas <= args.screen_feas_tol
    )


def screen_one(n: int, seed: int, args: argparse.Namespace) -> tuple[bool, dict[str, object]]:
    instance = generated_instance(n, seed, args)
    shor = solve_ttrs_relax(instance, method="Shor")
    row = result_row(instance, shor)
    selected = not is_rank_one_certified(row, args)
    return selected, {
        "n": n,
        "seed": seed,
        "generator": args.generator,
        "loga_radius": args.loga_radius,
        "screen_method": "Shor",
        "screen_status": shor.status,
        "screen_lower_bound": row.get("lower_bound"),
        "screen_upper_bound": row.get("upper_bound"),
        "screen_relative_certification_gap": row.get("relative_certification_gap"),
        "screen_relative_rank1_residual": row.get("relative_rank1_residual"),
        "screen_rank1_eig_ratio": row.get("rank1_eig_ratio"),
    }


def select_panel(args: argparse.Namespace) -> list[dict[str, object]]:
    selected_rows: list[dict[str, object]] = []
    for n in args.n:
        found = 0
        seed = 0
        while found < args.cases_per_size and seed < args.prefilter_max_seed:
            keep, screen = screen_one(n, seed, args)
            if keep:
                selected_rows.append(screen)
                found += 1
                if args.verbose:
                    print(
                        f"[select] n={n} seed={seed} kept "
                        f"cert={screen['screen_relative_certification_gap']} "
                        f"rank={screen['screen_relative_rank1_residual']}",
                        flush=True,
                    )
            elif args.verbose and seed > 0 and seed % args.prefilter_progress == 0:
                print(
                    f"[select] n={n} scanned {seed}; found "
                    f"{found}/{args.cases_per_size}",
                    flush=True,
                )
            seed += 1
        if found < args.cases_per_size:
            raise RuntimeError(
                f"only found {found} generated TTRS cases for n={n} "
                f"after scanning {args.prefilter_max_seed} seeds"
            )
    return selected_rows


def write_selected(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SELECTED_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_selected(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def existing_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {
            (row["file"], row["method"]): row
            for row in csv.DictReader(stream)
        }


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def add_generated_metadata(row: dict[str, object], selected: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    row = dict(row)
    row["reference_value"] = ""
    row["bound_gap"] = ""
    row["relative_bound_gap"] = ""
    if row.get("relative_certification_gap") in (None, ""):
        row["relative_certification_gap"] = relative_error(
            _float(row.get("upper_bound")), _float(row.get("lower_bound"))
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-panel-output", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--selected-panel-input", type=Path, default=None)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n", type=parse_dimensions, default=(20, 25, 30))
    parser.add_argument("--cases-per-size", type=int, default=10)
    parser.add_argument("--generator", choices=("centered",), default="centered")
    parser.add_argument("--loga-radius", type=float, default=1.0)
    parser.add_argument("--methods", default="lazy")
    parser.add_argument("--lazy-tol", type=float, default=1e-7)
    parser.add_argument("--lazy-max-iters", type=int, default=100)
    parser.add_argument("--lazy-cuts-per-iter", type=int, default=100)
    parser.add_argument(
        "--lazy-cut-strategy",
        choices=("svd", "coordinate"),
        default="coordinate",
        help="skew-skew separator for Lazy SEP; coordinate adds sparse "
        "individual equality cuts and is often more stable for hard large cases",
    )
    parser.add_argument("--screen-cert-tol", type=float, default=1e-5)
    parser.add_argument("--screen-rank-tol", type=float, default=1e-6)
    parser.add_argument("--screen-feas-tol", type=float, default=1e-7)
    parser.add_argument("--screen-eig-ratio-threshold", type=float, default=1e4)
    parser.add_argument("--prefilter-max-seed", type=int, default=10000)
    parser.add_argument("--prefilter-progress", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    if args.selected_panel_input is None:
        selected = select_panel(args)
        write_selected(args.selected_panel_output, selected)
        if args.verbose:
            print(f"wrote selected panel to {args.selected_panel_output}", flush=True)
    else:
        selected = read_selected(args.selected_panel_input)
    if args.select_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = existing_rows(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    skipped = 0
    with args.output.open(mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for index, chosen in enumerate(selected, start=1):
            n = int(chosen["n"])
            seed = int(chosen["seed"])
            instance = generated_instance(n, seed, args)
            for method in methods:
                key = (instance.name, method)
                if key in existing:
                    skipped += 1
                    continue
                if args.verbose:
                    print(
                        f"[{index}/{len(selected)}] run {instance.name} {method}",
                        flush=True,
                    )
                result = solve_ttrs_relax(
                    instance,
                    method=method,
                    lazy_tol=args.lazy_tol,
                    lazy_max_iters=args.lazy_max_iters,
                    lazy_cuts_per_iter=args.lazy_cuts_per_iter,
                    lazy_cut_strategy=args.lazy_cut_strategy,
                    verbose=args.verbose_solver,
                )
                row = add_generated_metadata(result_row(instance, result), chosen, args)
                writer.writerow(row)
                stream.flush()
                existing[key] = {k: str(v) for k, v in row.items()}
                written += 1
    if args.verbose:
        print(f"wrote {written} rows to {args.output} ({skipped} skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
