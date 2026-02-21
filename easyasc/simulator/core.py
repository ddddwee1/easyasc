from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from .. import globvars
from ._core_utils import validate_core_idx
from .cube import Cube
from .vec import Vec

if TYPE_CHECKING:
    from ..utils.instruction import Instruction


class Core:
    def __init__(self, core_idx: int) -> None:
        self.core_idx = validate_core_idx(core_idx)
        self.core_id = self.core_idx
        l1 = self._alloc_memory("l1_cap")
        ub1 = self._alloc_memory("ub_cap")
        ub2 = self._alloc_memory("ub_cap")
        self.cube = Cube(
            core_idx=self.core_idx,
            l1=l1,
            l0a=self._alloc_memory("l0a_cap"),
            l0b=self._alloc_memory("l0b_cap"),
            l0c=self._alloc_memory("l0c_cap"),
            ub1=ub1,
            ub2=ub2,
        )
        self.vecs: List[Vec] = [
            Vec(core_idx=self.core_idx, l1=l1, ub=ub1),
            Vec(core_idx=self.core_idx, l1=l1, ub=ub2),
        ]
        self.vec0 = self.vecs[0]
        self.vec1 = self.vecs[1]

    @staticmethod
    def _alloc_memory(cap_name: str) -> torch.Tensor:
        cap_value = getattr(globvars, cap_name, None)
        if not isinstance(cap_value, int) or cap_value < 0:
            raise ValueError(
                f"globvars.{cap_name} must be a non-negative int, got: {cap_value}"
            )
        return torch.empty((cap_value * 1024,), dtype=torch.uint8)

    def run(
        self,
        instructions: List["Instruction"],
        bound_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cube.run(instructions, bound_args=bound_args)
        for vec in self.vecs:
            vec.run(instructions, bound_args=bound_args)
