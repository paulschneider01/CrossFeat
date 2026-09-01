"""Modality aliases used by the minimal upload pipeline."""

from __future__ import annotations

_EQUIVALENCE_GROUPS: list[set[str]] = [
    {"rgb", "img", "opt", "optical"},
    {"cet1", "t1ce"},
    {"event", "events"},
]

# Build lookup: name -> frozenset of all equivalent names
_NAME_TO_GROUP: dict[str, frozenset[str]] = {}
for _group in _EQUIVALENCE_GROUPS:
    _frozen = frozenset(_group)
    for _name in _group:
        _NAME_TO_GROUP[_name] = _frozen

# Legacy single-hop alias dict kept for backward-compat with code that
# calls resolve_modality_alias() for a single canonical name.  The first
# entry in each equivalence group (alphabetically) is treated as canonical.
MODALITY_ALIASES: dict[str, str] = {}
for _group in _EQUIVALENCE_GROUPS:
    _canonical = sorted(_group)[0]
    for _name in _group:
        if _name != _canonical:
            MODALITY_ALIASES[_name] = _canonical

# Reverse mapping: canonical model name -> dataset-specific name.
_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in MODALITY_ALIASES.items()}


def get_equivalents(name: str) -> frozenset[str]:
    """Return all equivalent names for a modality, including itself.

    If the name is not in any equivalence group, returns ``{name}``.
    """
    return _NAME_TO_GROUP.get(name, frozenset({name}))


def resolve_modality_alias(name: str) -> str:
    """Resolve a modality name to its canonical alias, if one exists."""
    return MODALITY_ALIASES.get(name, name)


def resolve_reverse_alias(name: str) -> str:
    """Resolve a canonical model name back to a dataset-specific name.

    E.g., ``"cet1"`` -> ``"t1ce"`` for BRATS file lookup.
    Returns the input unchanged if no reverse alias exists.
    """
    return _REVERSE_ALIASES.get(name, name)


def find_modality_index(name: str, modality_to_idx: dict[str, int]) -> int | None:
    """Look up modality index, trying all equivalent names.

    Tries the literal *name* first, then every member of its equivalence
    class.  For namespaced keys (``"dataset.mod"``), the equivalence
    lookup is applied to the modality part only.

    Returns:
        The modality index, or None if no equivalent name is found.
    """
    # Direct hit
    if name in modality_to_idx:
        return modality_to_idx[name]

    # Extract bare modality (handle "dataset.mod" namespacing)
    if "." in name:
        ns, bare = name.split(".", 1)
    else:
        ns, bare = None, name

    # Try all equivalents
    for equiv in get_equivalents(bare):
        if equiv == bare:
            continue
        key = f"{ns}.{equiv}" if ns else equiv
        if key in modality_to_idx:
            return modality_to_idx[key]

    return None
