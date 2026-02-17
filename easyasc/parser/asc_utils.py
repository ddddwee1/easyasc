import re
from typing import Dict, Iterable, Set, Tuple

from ..utils.instruction import Instruction
from ..utils.positions import POSITION_CPP_MAPPING, Position
from ..utils.Tensor import DBuff, GMTensor, Tensor
from ..utils.var import Expr, Var

try:
    import sympy as sp
    from sympy.printing.str import StrPrinter
    from sympy.printing.precedence import precedence
except ImportError:
    sp = None
    StrPrinter = object  # type: ignore
    def precedence(expr):  # type: ignore
        return 0



_PRINTER = None

class _MulPowPrinter(StrPrinter):
    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()
        if exp.is_Integer and exp > 0:
            base_str = self.parenthesize(base, precedence(expr))
            return "*".join([base_str] * int(exp))
        return super()._print_Pow(expr) # type: ignore

    def _print_Mod(self, expr):
        lhs = self._print(expr.args[0]) # type: ignore
        rhs = self._print(expr.args[1]) # type: ignore
        return f"({lhs}) % ({rhs})"

    def _print_Equality(self, expr):
        lhs = self._print(expr.lhs) # type: ignore
        rhs = self._print(expr.rhs) # type: ignore
        return f"({lhs}) == ({rhs})"

    def _print_Unequality(self, expr):
        lhs = self._print(expr.lhs) # type: ignore
        rhs = self._print(expr.rhs) # type: ignore
        return f"({lhs}) != ({rhs})"

    def _print_And(self, expr):
        parts = [self._print(arg) for arg in expr.args] # type: ignore
        parts = ["(" + p + ")" for p in parts]
        return " && ".join(parts)

    def _print_Or(self, expr):
        parts = [self._print(arg) for arg in expr.args] # type: ignore
        parts = ["(" + p + ")" for p in parts]
        return " || ".join(parts)

_PRINTER = _MulPowPrinter() if sp is not None else None


def dtype_to_cpp(dtype) -> str:
    if dtype is None:
        return "auto"
    return str(dtype)


def position_to_cpp(position) -> str:
    key = str(position)
    if key not in POSITION_CPP_MAPPING:
        raise ValueError(f"未找到position映射: {key}")
    return POSITION_CPP_MAPPING[key]


def simplify_expr(expr: str) -> str:
    if sp is None or _PRINTER is None:
        return expr
    expr_norm = expr.replace(".", "___")
    try:
        expr_for_sympy = expr_norm.replace("&&", "&").replace("||", "|")
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_for_sympy))
        func_names = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr_for_sympy))
        locals_map = {}
        for name in names:
            if name == "true":
                locals_map[name] = sp.true
            elif name == "false":
                locals_map[name] = sp.false
            elif name in func_names:
                locals_map[name] = sp.Function(name)
            else:
                locals_map[name] = sp.Symbol(name)
        sym = sp.sympify(expr_for_sympy, locals=locals_map, evaluate=False)
        simplified = sp.simplify(sym, evaluate=False)
        result = _PRINTER.doprint(simplified)
    except Exception:
        return expr
    return str(result).replace("___", ".")


def _needs_parens(expr: str) -> bool:
    return any(token in expr for token in (" + ", " - ", " * ", " / "))


def _wrap_expr(expr: str) -> str:
    if _needs_parens(expr):
        return f"({expr})"
    return expr


def _is_zero_literal(expr: str) -> bool:
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text == "0"


def format_binop(op: str, left: str, right: str) -> str:
    return f"{_wrap_expr(left)} {op} {_wrap_expr(right)}"


def is_tmp_var(value: object) -> bool:
    return isinstance(value, Var) and value.name.startswith("_tmp_var_")


def is_tmp_tensor(value: object) -> bool:
    return isinstance(value, Tensor) and value.name.startswith("_tmp_tensor_")


def is_tmp_gmtensor(value: object) -> bool:
    return isinstance(value, GMTensor) and value.name.startswith("_tmp_gmtensor_")


def value_to_cpp(value, expr_map: Dict[str, str]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Expr):
        expr = str(value)
        if expr_map:
            for name, mapped in expr_map.items():
                if name not in expr:
                    continue
                mapped_expr = simplify_expr(mapped)
                mapped_expr = _wrap_expr(mapped_expr)
                expr = re.sub(rf"\b{re.escape(name)}\b", mapped_expr, expr)
        return expr
    if isinstance(value, Var):
        mapped = expr_map.get(value.name)
        if mapped is not None:
            simplified = simplify_expr(mapped)
            if simplified != mapped:
                expr_map[value.name] = simplified
            return simplified
        return value.name
    if isinstance(value, Tensor):
        mapped = expr_map.get(value.name)
        if mapped is not None:
            return mapped
        return value.name
    if isinstance(value, GMTensor):
        mapped = expr_map.get(value.name)
        if mapped is not None:
            return mapped
        return value.name
    if isinstance(value, DBuff):
        return value.name
    return str(value)


def build_offset_expr(shape, offset, expr_map: Dict[str, str]) -> str:
    if shape is None or offset is None:
        raise ValueError("slice_gm_tensor需要out包含shape与offset")
    if len(shape) != len(offset):
        raise ValueError("slice_gm_tensor的shape与offset维度不一致")
    if not shape:
        return "0"
    # Linearized offset: sum(off[i] * prod(shape[i+1:])) + off[last]
    terms = []
    dim = len(shape)
    for idx in range(dim - 1):
        off_raw = value_to_cpp(offset[idx], expr_map)
        if _is_zero_literal(off_raw):
            continue
        off = _wrap_expr(off_raw)
        if dim - idx - 1 == 1:
            stride_expr = _wrap_expr(value_to_cpp(shape[idx + 1], expr_map))
        else:
            strides = [_wrap_expr(value_to_cpp(shape[j], expr_map)) for j in range(idx + 1, dim)]
            stride_expr = " * ".join(strides)
        terms.append(f"{off} * {stride_expr}")
    tail = value_to_cpp(offset[-1], expr_map)
    if not _is_zero_literal(tail):
        terms.append(_wrap_expr(tail))
    if not terms:
        return "0"
    return " + ".join(terms)


def build_offset_expr_nz(shape, offset, dtype, expr_map: Dict[str, str]) -> str:
    if shape is None or offset is None:
        raise ValueError("slice_tensor需要out包含shape与offset")
    if len(shape) != 2 or len(offset) != 2:
        raise ValueError("slice_tensor的shape与offset必须为2维")
    if dtype is None:
        raise ValueError("slice_tensor需要out包含dtype")
    off0 = _wrap_expr(value_to_cpp(offset[0], expr_map))
    off1 = _wrap_expr(value_to_cpp(offset[1], expr_map))
    shape0 = _wrap_expr(value_to_cpp(shape[0], expr_map))
    expr = f"{off0} * {dtype.C0} + {off1} * {shape0}"
    return expr


_ASSIGNMENT_OPS = {
    "GetCubeNum",
    "GetCubeIdx",
    "GetVecNum",
    "GetVecIdx",
    "GetSubBlockIdx",
    "scalar_sqrt",
    "Align16",
    "Align32",
    "Align64",
    "Align128",
    "Align256",
    "CeilDiv",
    "Min",
    "Max",
    "var_mul",
    "var_div",
    "var_add",
    "var_sub",
}


def is_assignment_op(opname: str) -> bool:
    return opname in _ASSIGNMENT_OPS


def uses_var_in_operands(inst: Instruction, name: str) -> bool:
    if inst.opname in ("GetCubeNum", "GetCubeIdx", "GetVecNum", "GetVecIdx", "GetSubBlockIdx"):
        return False
    if inst.opname in (
        "CeilDiv",
        "Min",
        "Max",
        "var_mul",
        "var_div",
        "var_add",
        "var_sub",
        "scalar_sqrt",
        "Align16",
        "Align32",
        "Align64",
        "Align128",
        "Align256",
    ):
        for key in ("a", "b"):
            val = inst.kwargs.get(key, None)
            if isinstance(val, Var) and val.name == name:
                return True
    return False


def assignment_expr(inst: Instruction, expr_map: Dict[str, str]) -> str:
    opname = inst.opname
    if opname == "GetCubeNum":
        return "GetBlockNum()"
    if opname == "GetCubeIdx":
        return "get_block_idx()"
    if opname == "GetVecNum":
        return "GetBlockNum() * 2"
    if opname == "GetVecIdx":
        return "GetBlockIdx()"
    if opname == "GetSubBlockIdx":
        return "get_subblockid()"
    if opname == "scalar_sqrt":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"sqrt({a})"
    if opname == "Align16":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"Align16B({a})"
    if opname == "Align32":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"Align32B({a})"
    if opname == "Align64":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"Align64B({a})"
    if opname == "Align128":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"Align128B({a})"
    if opname == "Align256":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        return f"Align256B({a})"
    if opname == "CeilDiv":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return f"CeilDiv({a}, {b})"
    if opname == "Min":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return f"Min({a}, {b})"
    if opname == "Max":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return f"Max({a}, {b})"
    if opname == "var_mul":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return simplify_expr(format_binop("*", a, b))
    if opname == "var_div":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return simplify_expr(format_binop("/", a, b))
    if opname == "var_add":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return simplify_expr(format_binop("+", a, b))
    if opname == "var_sub":
        a = value_to_cpp(inst.kwargs.get("a", None), expr_map)
        b = value_to_cpp(inst.kwargs.get("b", None), expr_map)
        return simplify_expr(format_binop("-", a, b))
    raise ValueError(f"unsupported assignment op: {opname}")


def build_expr_state(
    instructions: Iterable[Instruction],
) -> Tuple[Dict[str, str], Set[str], Set[str], Set[str]]:
    expr_map: Dict[str, str] = {}
    tmp_var_names: Set[str] = set()
    for inst in instructions:
        opname = inst.opname
        if opname == "GetCubeNum":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and is_tmp_var(out):
                expr_map[out.name] = "GetBlockNum()"
                tmp_var_names.add(out.name)
            continue
        if opname == "GetCubeIdx":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and is_tmp_var(out):
                expr_map[out.name] = "get_block_idx()"
                tmp_var_names.add(out.name)
            continue
        if opname == "GetVecNum":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and is_tmp_var(out):
                expr_map[out.name] = "GetBlockNum() * 2"
                tmp_var_names.add(out.name)
            continue
        if opname == "GetVecIdx":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and is_tmp_var(out):
                expr_map[out.name] = "GetBlockIdx()"
                tmp_var_names.add(out.name)
            continue
        if opname == "GetSubBlockIdx":
            out = inst.kwargs.get("out", None)
            if isinstance(out, Var) and is_tmp_var(out):
                expr_map[out.name] = "get_subblockid()"
                tmp_var_names.add(out.name)
            continue
        if opname in ("CeilDiv", "Min", "Max", "var_mul", "var_div", "var_add", "var_sub"):
            out = inst.kwargs.get("out", None)
            if not isinstance(out, Var) or not is_tmp_var(out):
                continue
            expr_map[out.name] = assignment_expr(inst, expr_map)
            tmp_var_names.add(out.name)
        if opname in ("scalar_sqrt", "Align16", "Align32", "Align64", "Align128", "Align256"):
            out = inst.kwargs.get("out", None)
            if not isinstance(out, Var) or not is_tmp_var(out):
                continue
            expr_map[out.name] = assignment_expr(inst, expr_map)
            tmp_var_names.add(out.name)

    tmp_tensor_names: Set[str] = set()
    for inst in instructions:
        if inst.opname == "get_buf":
            out = inst.kwargs.get("out", None)
            if not isinstance(out, Tensor) or not is_tmp_tensor(out):
                continue
            buf = inst.kwargs.get("buf", None)
            if not isinstance(buf, DBuff):
                continue
            index = value_to_cpp(inst.kwargs.get("index", None), expr_map)
            expr_map[out.name] = f"{buf.name}.get({index})"
            tmp_tensor_names.add(out.name)
            continue
        if inst.opname == "slice_tensor":
            out = inst.kwargs.get("out", None)
            if not isinstance(out, Tensor) or not is_tmp_tensor(out):
                continue
            src = inst.kwargs.get("src", None)
            shape = getattr(out, "shape", None)
            offset = getattr(out, "offset", None)
            if out.position is Position.L1:
                offset_expr = simplify_expr(build_offset_expr_nz(shape, offset, out.dtype, expr_map))
            else:
                offset_expr = simplify_expr(build_offset_expr(shape, offset, expr_map))
            src_expr = value_to_cpp(src, expr_map)
            if offset_expr == "0":
                expr_map[out.name] = f"{src_expr}"
            else:
                expr_map[out.name] = f"{src_expr}[{offset_expr}]"
            tmp_tensor_names.add(out.name)
            continue
        if inst.opname == "micro_slice_tensor":
            out = inst.kwargs.get("out", None)
            if not isinstance(out, Tensor) or not is_tmp_tensor(out):
                continue
            src = inst.kwargs.get("src", None)
            shape = getattr(out, "shape", None)
            offset = getattr(out, "offset", None)
            offset_expr = simplify_expr(build_offset_expr(shape, offset, expr_map))
            src_expr = value_to_cpp(src, expr_map)
            if offset_expr == "0":
                expr_map[out.name] = f"{src_expr}"
            else:
                expr_map[out.name] = f"{src_expr} + {offset_expr}"
            tmp_tensor_names.add(out.name)
            continue

    tmp_gmtensor_names: Set[str] = set()
    for inst in instructions:
        if inst.opname != "slice_gm_tensor":
            continue
        out = inst.kwargs.get("out", None)
        if not isinstance(out, GMTensor) or not is_tmp_gmtensor(out):
            continue
        src = inst.kwargs.get("src", None)
        shape = getattr(out, "shape", None)
        offset = getattr(out, "offset", None)
        offset_expr = simplify_expr(build_offset_expr(shape, offset, expr_map))
        src_expr = value_to_cpp(src, expr_map)
        if offset_expr == "0":
            expr_map[out.name] = f"{src_expr}"
        else:
            expr_map[out.name] = f"{src_expr}[{offset_expr}]"
        tmp_gmtensor_names.add(out.name)

    return expr_map, tmp_var_names, tmp_tensor_names, tmp_gmtensor_names


def should_skip_inst(
    inst: Instruction,
    tmp_var_names: Set[str],
    tmp_tensor_names: Set[str],
    tmp_gmtensor_names: Set[str],
) -> bool:
    if inst.opname == "create_var":
        val = inst.kwargs.get("val", None)
        return isinstance(val, Var) and is_tmp_var(val) and val.name in tmp_var_names
    if inst.opname in (
        "GetCubeNum",
        "GetCubeIdx",
        "GetVecNum",
        "GetVecIdx",
        "GetSubBlockIdx",
        "scalar_sqrt",
        "Align16",
        "Align32",
        "Align64",
        "Align128",
        "Align256",
        "CeilDiv",
        "Min",
        "Max",
        "var_mul",
        "var_div",
        "var_add",
        "var_sub",
    ):
        out = inst.kwargs.get("out", None)
        return isinstance(out, Var) and out.name in tmp_var_names
    if inst.opname in ("get_buf", "slice_tensor", "micro_slice_tensor", "create_tensor"):
        out = inst.kwargs.get("out", None)
        return isinstance(out, Tensor) and is_tmp_tensor(out) and out.name in tmp_tensor_names
    if inst.opname == "slice_gm_tensor":
        out = inst.kwargs.get("out", None)
        return isinstance(out, GMTensor) and is_tmp_gmtensor(out) and out.name in tmp_gmtensor_names
    return False
