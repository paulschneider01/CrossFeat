"""Case container type used across train/eval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CaseData:
    """In-memory case representation for a modality-aligned sample."""

    case_id: str
    volumes: Dict[str, np.ndarray] = field(default_factory=dict)
    affines: Dict[str, np.ndarray] = field(default_factory=dict)
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> List[str]:
        return list(self.volumes.keys())

    def get_common_shape(self) -> Optional[Tuple[int, ...]]:
        shapes = list(self.shapes.values())
        if len(set(shapes)) == 1:
            return shapes[0]
        return None
