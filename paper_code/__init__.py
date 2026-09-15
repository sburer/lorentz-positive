"""Minimal paper-code package for the manuscript's computational sections."""

from .instances import BilinearInstance, generate_instance, paper_panel
# Both modules define generate_named_instance; re-export the Section 4.1 one
# under an unambiguous name.
from .quad_instances import QuadInstance, maxdist_panel
from .quad_instances import generate_named_instance as generate_maxdist_instance

__all__ = [
    "BilinearInstance",
    "generate_instance",
    "paper_panel",
    "QuadInstance",
    "generate_maxdist_instance",
    "maxdist_panel",
]
