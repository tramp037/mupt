"""
Canonical roles that Primitives can play in an exportable hierarchy.

These roles allow Primitives to be explicitly tagged with their semantic
purpose within a molecular representation, enabling generic tree traversal
for export to external toolkits (e.g., MDAnalysis, RDKit) without
hard-coding assumptions about tree depth or structure.
"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from enum import Enum


class PrimitiveRole(Enum):
    """Canonical roles that Primitives can play in an exportable hierarchy.
    
    These roles map to the standard levels expected by molecular analysis
    toolkits such as MDAnalysis:
    
    - UNASSIGNED: No role has been assigned (default)
    - UNIVERSE:  Root container of the entire system
    - SEGMENT:   Non-covalently bonded entity (chain, molecule)
    - RESIDUE:   Repeating sub-unit (monomer, amino acid, CG bead group)
    - PARTICLE:  Exportable particle (atom in all-atom, bead in CG)

    Primitives at intermediate depths between role-tagged levels should
    use ``UNASSIGNED`` to indicate transparent grouping nodes.
    
    Examples
    --------
    >>> from mupt.mupr.roles import PrimitiveRole
    >>> PrimitiveRole.UNASSIGNED
    <PrimitiveRole.UNASSIGNED: 'unassigned'>
    >>> PrimitiveRole.UNIVERSE is not PrimitiveRole.UNASSIGNED
    True
    """
    UNASSIGNED = "unassigned"
    UNIVERSE = "universe"
    SEGMENT  = "segment"
    RESIDUE  = "residue"
    PARTICLE = "particle"
