from typing import Callable, Dict

from ..helper import CodeHelper
from ..asc_utils import (
    build_offset_expr,
    build_offset_expr_nz,
    dtype_to_cpp,
    format_binop,
    position_to_cpp,
    value_to_cpp,
)
from ...utils.instruction import Instruction
from ...utils.Tensor import DBuff, GMTensor, Tensor
from ...utils.events import SEvent, DEvent
from ...utils.positions import Position
from ...utils.pipe import PipeType
from ...utils.var import Var


Handler = Callable[[Instruction, CodeHelper, Dict[str, str]], None]


def _pipe_name(pipe) -> str:
    return f"PIPE_{pipe}"


def handle_start_auto_sync(inst: Instruction, helper: CodeHelper, expr_map: Dict[str, str]) -> None:
    # helper("// start auto sync")
    ...


def handle_end_auto_sync(inst: Instruction, helper: CodeHelper, expr_map: Dict[str, str]) -> None:
    # helper("// end auto sync")
    ...
