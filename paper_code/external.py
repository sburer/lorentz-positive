"""Locate the companion ``ballconstraints`` checkout.

The 212 TTRS instances of Section 4.2 and the beta relaxation used by the
Section 4.2 counterexample live in https://github.com/sburer/ballconstraints.
That repository is vendored as the git submodule
``external/ballconstraints`` (pinned to a fixed commit); run
``git submodule update --init`` once after cloning.  The environment variable
``BALLCONSTRAINTS_DIR`` overrides the location, and a sibling checkout next to
this repository is accepted as a last resort.
"""

from __future__ import annotations

import os
from pathlib import Path

PAPER_CODE_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = PAPER_CODE_ROOT / "external" / "ballconstraints"

CLONE_HINT = (
    "ballconstraints checkout not found.  Run\n"
    "    git submodule update --init\n"
    "from the repository root, or set BALLCONSTRAINTS_DIR to a checkout of\n"
    "https://github.com/sburer/ballconstraints."
)


def ballconstraints_root() -> Path:
    """Return the ballconstraints checkout, raising a helpful error if absent."""

    candidates: list[Path] = []
    override = os.environ.get("BALLCONSTRAINTS_DIR")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(SUBMODULE)
    # Sibling checkouts: next to this directory, and next to a parent
    # repository that contains this directory.
    candidates.append(PAPER_CODE_ROOT.parent / "ballconstraints")
    candidates.append(PAPER_CODE_ROOT.parent.parent / "ballconstraints")
    for candidate in candidates:
        if (candidate / "src" / "define_functions.py").exists():
            return candidate
    raise FileNotFoundError(
        CLONE_HINT + "\nLooked in: " + ", ".join(str(c) for c in candidates)
    )


def ttrs_instance_dir() -> Path:
    """Directory holding the 212 Section 5.3 ``instance_*.mat`` files."""

    return (
        ballconstraints_root()
        / "data"
        / "soctrust"
        / "Section_5_3_unsolved_instances_saved"
    )
