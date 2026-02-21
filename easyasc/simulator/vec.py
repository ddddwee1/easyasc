from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from ._core_utils import validate_core_idx

if TYPE_CHECKING:
    from ..utils.instruction import Instruction


class Vec:
    def __init__(self, core_idx: int, l1: torch.Tensor, ub: torch.Tensor) -> None:
        self.core_idx = validate_core_idx(core_idx)
        self.L1 = l1
        self.UB = ub

    def run(
        self,
        instructions: List["Instruction"],
        bound_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ = instructions
        _ = bound_args
        pass
