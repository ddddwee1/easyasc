from typing import Dict

from .common import Handler, Tensor, dtype_to_cpp, value_to_cpp
from ...utils.reg import Reg, MaskReg
from ...utils.castconfig import CastConfig
from ...utils.comparemode import CompareModeType


def _require_reg(value, label: str) -> Reg:
    if not isinstance(value, Reg):
        raise TypeError(f"{label}requires Reg type, current type: {type(value)}")
    return value


def _require_mask(value, label: str) -> MaskReg:
    if not isinstance(value, MaskReg):
        raise TypeError(f"{label}requires MaskReg type, current type: {type(value)}")
    return value


def _require_tensor(value, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label}requires Tensor type, current type: {type(value)}")
    return value


def _require_str(value, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label}requires strtype, current type: {type(value)}")
    return value


def _mode_name(mode) -> str:
    if isinstance(mode, CompareModeType):
        return mode.name
    return str(mode)


def _handle_micro_unary(inst, helper, expr_map, func_name: str) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), func_name)
    src = _require_reg(inst.kwargs.get("src", None), func_name)
    mask = _require_mask(inst.kwargs.get("mask", None), func_name)

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::{func_name}({dst_expr}, {src_expr}, {mask_expr});")


def _handle_micro_binary(inst, helper, expr_map, func_name: str) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), func_name)
    src1 = _require_reg(inst.kwargs.get("src1", None), func_name)
    src2 = _require_reg(inst.kwargs.get("src2", None), func_name)
    mask = _require_mask(inst.kwargs.get("mask", None), func_name)

    dst_expr = value_to_cpp(dst, expr_map)
    src1_expr = value_to_cpp(src1, expr_map)
    src2_expr = value_to_cpp(src2, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::{func_name}({dst_expr}, {src1_expr}, {src2_expr}, {mask_expr});")


def _handle_micro_scalar(inst, helper, expr_map, func_name: str) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), func_name)
    src = _require_reg(inst.kwargs.get("src", None), func_name)
    mask = _require_mask(inst.kwargs.get("mask", None), func_name)

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    v_expr = value_to_cpp(inst.kwargs.get("v", None), expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::{func_name}({dst_expr}, {src_expr}, {v_expr}, {mask_expr});")


def handle_micro_ub2reg(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "UB2REG")
    src = _require_tensor(inst.kwargs.get("src", None), "UB2REG")
    mask = _require_mask(inst.kwargs.get("mask", None), "UB2REG")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    blk_stride = value_to_cpp(inst.kwargs.get("blk_stride", None), expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"MicroAPI::DataCopy<{dtype}, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({dst_expr}, {src_expr}, {blk_stride}, {mask_expr});"
    )


def handle_micro_reg2ub(inst, helper, expr_map) -> None:
    dst = _require_tensor(inst.kwargs.get("dst", None), "REG2UB")
    src = _require_reg(inst.kwargs.get("src", None), "REG2UB")
    mask = _require_mask(inst.kwargs.get("mask", None), "REG2UB")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    blk_stride = value_to_cpp(inst.kwargs.get("blk_stride", None), expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    dtype = dtype_to_cpp(src.dtype)
    helper(
        f"MicroAPI::DataCopy<{dtype}, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>({dst_expr}, {src_expr}, {blk_stride}, {mask_expr});"
    )


def handle_micro_ub2regcont(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "UB2REGCONT")
    src = _require_tensor(inst.kwargs.get("src", None), "UB2REGCONT")
    mode = _require_str(inst.kwargs.get("mode", None), "UB2REGCONT")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"MicroAPI::DataCopy<{dtype}, MicroAPI::LoadDist::{mode}>({dst_expr}, {src_expr});"
    )


def handle_micro_reg2ubcont(inst, helper, expr_map) -> None:
    dst = _require_tensor(inst.kwargs.get("dst", None), "REG2UBCONT")
    src = _require_reg(inst.kwargs.get("src", None), "REG2UBCONT")
    mask = _require_mask(inst.kwargs.get("mask", None), "REG2UBCONT")
    mode = _require_str(inst.kwargs.get("mode", None), "REG2UBCONT")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    dtype = dtype_to_cpp(src.dtype)
    helper(
        f"MicroAPI::DataCopy<{dtype}, MicroAPI::StoreDist::{mode}>({dst_expr}, {src_expr}, {mask_expr});"
    )


def handle_micro_vexp(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Exp")


def handle_micro_vabs(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Abs")


def handle_micro_vrelu(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Relu")


def handle_micro_vsqrt(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Sqrt")


def handle_micro_vln(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Ln")


def handle_micro_vlog(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Log")


def handle_micro_vlog2(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Log2")


def handle_micro_vlog10(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Log10")


def handle_micro_vneg(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Neg")


def handle_micro_vnot(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Not")


def handle_micro_vcopy(inst, helper, expr_map) -> None:
    _handle_micro_unary(inst, helper, expr_map, "Copy")


def handle_micro_vmax(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Max")


def handle_micro_vmin(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Min")


def handle_micro_vadd(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Add")


def handle_micro_vsub(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Sub")


def handle_micro_vmul(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Mul")


def handle_micro_vdiv(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Div")


def handle_micro_vand(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "And")


def handle_micro_vor(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Or")


def handle_micro_vxor(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Xor")


def handle_micro_vprelu(inst, helper, expr_map) -> None:
    _handle_micro_binary(inst, helper, expr_map, "Prelu")


def handle_micro_vmaxs(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "Maxs")


def handle_micro_vmins(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "Mins")


def handle_micro_vadds(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "Adds")


def handle_micro_vmuls(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "Muls")


def handle_micro_vlrelu(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "LeakyRelu")


def handle_micro_shiftls(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "SHIFTLS")
    src = _require_reg(inst.kwargs.get("src", None), "SHIFTLS")
    mask = _require_mask(inst.kwargs.get("mask", None), "SHIFTLS")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    v_expr = value_to_cpp(inst.kwargs.get("v", None), expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ShiftLefts({dst_expr}, {src_expr}, (int16_t){v_expr}, {mask_expr});")


def handle_micro_shiftrs(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "SHIFTRS")
    src = _require_reg(inst.kwargs.get("src", None), "SHIFTRS")
    mask = _require_mask(inst.kwargs.get("mask", None), "SHIFTRS")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    v_expr = value_to_cpp(inst.kwargs.get("v", None), expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ShiftRights({dst_expr}, {src_expr}, (int16_t){v_expr}, {mask_expr});")


def handle_micro_vaxpy(inst, helper, expr_map) -> None:
    _handle_micro_scalar(inst, helper, expr_map, "Axpy")


def handle_micro_vcadd(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCADD")
    src = _require_reg(inst.kwargs.get("src", None), "VCADD")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCADD")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceSum({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcmax(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCMAX")
    src = _require_reg(inst.kwargs.get("src", None), "VCMAX")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCMAX")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceMax({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcmin(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCMIN")
    src = _require_reg(inst.kwargs.get("src", None), "VCMIN")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCMIN")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceMin({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcgadd(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCGADD")
    src = _require_reg(inst.kwargs.get("src", None), "VCGADD")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCGADD")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceSumWithDataBlock({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcgmax(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCGMAX")
    src = _require_reg(inst.kwargs.get("src", None), "VCGMAX")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCGMAX")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceMaxWithDataBlock({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcgmin(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCGMIN")
    src = _require_reg(inst.kwargs.get("src", None), "VCGMIN")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCGMIN")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::ReduceMinWithDataBlock({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vcpadd(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VCPADD")
    src = _require_reg(inst.kwargs.get("src", None), "VCPADD")
    mask = _require_mask(inst.kwargs.get("mask", None), "VCPADD")
    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::PairReduceSum({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_vdup(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "VDUP")
    mask = _require_mask(inst.kwargs.get("mask", None), "VDUP")
    src_val = inst.kwargs.get("src", None)
    if src_val is None:
        raise TypeError("VDUP requires srcparameter")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src_val, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    helper(f"MicroAPI::Duplicate({dst_expr}, {src_expr}, {mask_expr});")


def handle_micro_dinterleave(inst, helper, expr_map) -> None:
    dst0 = _require_reg(inst.kwargs.get("dst0", None), "DINTERLEAVE")
    dst1 = _require_reg(inst.kwargs.get("dst1", None), "DINTERLEAVE")
    src0 = _require_reg(inst.kwargs.get("src0", None), "DINTERLEAVE")
    src1 = _require_reg(inst.kwargs.get("src1", None), "DINTERLEAVE")
    helper(
        f"MicroAPI::DeInterleave({value_to_cpp(dst0, expr_map)}, {value_to_cpp(dst1, expr_map)}, "
        f"{value_to_cpp(src0, expr_map)}, {value_to_cpp(src1, expr_map)});"
    )


def handle_micro_interleave(inst, helper, expr_map) -> None:
    dst0 = _require_reg(inst.kwargs.get("dst0", None), "INTERLEAVE")
    dst1 = _require_reg(inst.kwargs.get("dst1", None), "INTERLEAVE")
    src0 = _require_reg(inst.kwargs.get("src0", None), "INTERLEAVE")
    src1 = _require_reg(inst.kwargs.get("src1", None), "INTERLEAVE")
    helper(
        f"MicroAPI::Interleave({value_to_cpp(dst0, expr_map)}, {value_to_cpp(dst1, expr_map)}, "
        f"{value_to_cpp(src0, expr_map)}, {value_to_cpp(src1, expr_map)});"
    )


def handle_micro_cast(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "CAST")
    src = _require_reg(inst.kwargs.get("src", None), "CAST")
    mask = _require_mask(inst.kwargs.get("mask", None), "CAST")
    ddst = inst.kwargs.get("ddst", None)
    dsrc = inst.kwargs.get("dsrc", None)
    config = inst.kwargs.get("config", None)
    if not isinstance(config, CastConfig):
        raise TypeError(f"CAST requires CastConfigtype, current type: {type(config)}")

    dst_expr = value_to_cpp(dst, expr_map)
    src_expr = value_to_cpp(src, expr_map)
    mask_expr = value_to_cpp(mask, expr_map)
    ddst_cpp = dtype_to_cpp(ddst)
    dsrc_cpp = dtype_to_cpp(dsrc)
    helper(
        f"MicroAPI::Cast<{ddst_cpp}, {dsrc_cpp}, {config}>({dst_expr}, {src_expr}, {mask_expr});"
    )


def handle_micro_arange(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "ARANGE")
    dtype = inst.kwargs.get("dtype", None)
    mode = _require_str(inst.kwargs.get("mode", None), "ARANGE")
    v = inst.kwargs.get("v", None)
    if v is None:
        raise TypeError("ARANGE requires vparameter")

    dst_expr = value_to_cpp(dst, expr_map)
    v_expr = value_to_cpp(v, expr_map)
    dtype_cpp = dtype_to_cpp(dtype)
    helper(f"MicroAPI::Arange<{dtype_cpp}, {mode}>({dst_expr}, {v_expr});")


def handle_micro_datacopygather(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "DATACOPYGATHER")
    src = _require_tensor(inst.kwargs.get("src", None), "DATACOPYGATHER")
    index = _require_reg(inst.kwargs.get("index", None), "DATACOPYGATHER")
    mask = _require_mask(inst.kwargs.get("mask", None), "DATACOPYGATHER")
    helper(
        f"MicroAPI::DataCopyGather({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, "
        f"{value_to_cpp(index, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_datacopyscatter(inst, helper, expr_map) -> None:
    dst = _require_tensor(inst.kwargs.get("dst", None), "DATACOPYSCATTER")
    src = _require_reg(inst.kwargs.get("src", None), "DATACOPYSCATTER")
    index = _require_reg(inst.kwargs.get("index", None), "DATACOPYSCATTER")
    mask = _require_mask(inst.kwargs.get("mask", None), "DATACOPYSCATTER")
    helper(
        f"MicroAPI::DataCopyScatter({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, "
        f"{value_to_cpp(index, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_gather(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "GATHER")
    src = _require_reg(inst.kwargs.get("src", None), "GATHER")
    index = _require_reg(inst.kwargs.get("index", None), "GATHER")
    helper(
        f"MicroAPI::Gather({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, {value_to_cpp(index, expr_map)});"
    )


def handle_micro_gathermask(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "GATHERMASK")
    src = _require_reg(inst.kwargs.get("src", None), "GATHERMASK")
    mask = _require_mask(inst.kwargs.get("mask", None), "GATHERMASK")
    helper(
        f"MicroAPI::GatherMask({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_compare(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "COMPARE")
    src1 = _require_reg(inst.kwargs.get("src1", None), "COMPARE")
    src2 = _require_reg(inst.kwargs.get("src2", None), "COMPARE")
    dtype = inst.kwargs.get("dtype", None)
    mode = inst.kwargs.get("mode", None)
    mask = _require_mask(inst.kwargs.get("mask", None), "COMPARE")

    dtype_cpp = dtype_to_cpp(dtype)
    mode_name = _mode_name(mode)
    helper(
        f"MicroAPI::Compare<{dtype_cpp}, CMPMODE::{mode_name}>({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, "
        f"{value_to_cpp(src2, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_compares(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "COMPARES")
    src1 = _require_reg(inst.kwargs.get("src1", None), "COMPARES")
    src2 = inst.kwargs.get("src2", None)
    dtype = inst.kwargs.get("dtype", None)
    mode = inst.kwargs.get("mode", None)
    mask = _require_mask(inst.kwargs.get("mask", None), "COMPARES")

    dtype_cpp = dtype_to_cpp(dtype)
    mode_name = _mode_name(mode)
    helper(
        f"MicroAPI::CompareScalar<{dtype_cpp}, CMPMODE::{mode_name}>({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, "
        f"{value_to_cpp(src2, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_select(inst, helper, expr_map) -> None:
    dst = _require_reg(inst.kwargs.get("dst", None), "SELECT")
    src1 = _require_reg(inst.kwargs.get("src1", None), "SELECT")
    src2 = _require_reg(inst.kwargs.get("src2", None), "SELECT")
    mask = _require_mask(inst.kwargs.get("mask", None), "SELECT")
    helper(
        f"MicroAPI::Select({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, {value_to_cpp(src2, expr_map)}, "
        f"{value_to_cpp(mask, expr_map)});"
    )


def handle_micro_maskand(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKAND")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKAND")
    src2 = _require_mask(inst.kwargs.get("src2", None), "MASKAND")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKAND")
    helper(
        f"MicroAPI::MaskAnd({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, {value_to_cpp(src2, expr_map)}, "
        f"{value_to_cpp(mask, expr_map)});"
    )


def handle_micro_maskor(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKOR")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKOR")
    src2 = _require_mask(inst.kwargs.get("src2", None), "MASKOR")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKOR")
    helper(
        f"MicroAPI::MaskOr({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, {value_to_cpp(src2, expr_map)}, "
        f"{value_to_cpp(mask, expr_map)});"
    )


def handle_micro_maskxor(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKXOR")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKXOR")
    src2 = _require_mask(inst.kwargs.get("src2", None), "MASKXOR")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKXOR")
    helper(
        f"MicroAPI::MaskXor({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, {value_to_cpp(src2, expr_map)}, "
        f"{value_to_cpp(mask, expr_map)});"
    )


def handle_micro_masknot(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKNOT")
    src = _require_mask(inst.kwargs.get("src", None), "MASKNOT")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKNOT")
    helper(
        f"MicroAPI::MaskNot({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_maskmov(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKMOV")
    src = _require_mask(inst.kwargs.get("src", None), "MASKMOV")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKMOV")
    helper(
        f"MicroAPI::MaskMov({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)}, {value_to_cpp(mask, expr_map)});"
    )


def handle_micro_maskinterl(inst, helper, expr_map) -> None:
    dst0 = _require_mask(inst.kwargs.get("dst0", None), "MASKINTERL")
    dst1 = _require_mask(inst.kwargs.get("dst1", None), "MASKINTERL")
    src0 = _require_mask(inst.kwargs.get("src0", None), "MASKINTERL")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKINTERL")
    helper(
        f"MicroAPI::MaskInterleave({value_to_cpp(dst0, expr_map)}, {value_to_cpp(dst1, expr_map)}, "
        f"{value_to_cpp(src0, expr_map)}, {value_to_cpp(src1, expr_map)});"
    )


def handle_micro_maskdeinterl(inst, helper, expr_map) -> None:
    dst0 = _require_mask(inst.kwargs.get("dst0", None), "MASKDEINTERL")
    dst1 = _require_mask(inst.kwargs.get("dst1", None), "MASKDEINTERL")
    src0 = _require_mask(inst.kwargs.get("src0", None), "MASKDEINTERL")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKDEINTERL")
    helper(
        f"MicroAPI::MaskDeInterleave({value_to_cpp(dst0, expr_map)}, {value_to_cpp(dst1, expr_map)}, "
        f"{value_to_cpp(src0, expr_map)}, {value_to_cpp(src1, expr_map)});"
    )


def handle_micro_masksel(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKSEL")
    src1 = _require_mask(inst.kwargs.get("src1", None), "MASKSEL")
    src2 = _require_mask(inst.kwargs.get("src2", None), "MASKSEL")
    mask = _require_mask(inst.kwargs.get("mask", None), "MASKSEL")
    helper(
        f"MicroAPI::MaskSel({value_to_cpp(dst, expr_map)}, {value_to_cpp(src1, expr_map)}, {value_to_cpp(src2, expr_map)}, "
        f"{value_to_cpp(mask, expr_map)});"
    )


def handle_micro_movemaskspr(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MOVEMASKSPR")
    dtype = dtype_to_cpp(dst.dtype)
    helper(f"{value_to_cpp(dst, expr_map)} = MicroAPI::MoveMask<{dtype}>();")


def handle_micro_updatemask(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "UPDATEMASK")
    cnt = inst.kwargs.get("cnt", None)
    if cnt is None:
        raise TypeError("UPDATEMASK requires cntparameter")
    dtype = dtype_to_cpp(dst.dtype)
    helper(
        f"{value_to_cpp(dst, expr_map)} = MicroAPI::UpdateMask<{dtype}>({value_to_cpp(cnt, expr_map)});"
    )


def handle_micro_maskpack(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKPACK")
    src = _require_mask(inst.kwargs.get("src", None), "MASKPACK")
    mode = _require_str(inst.kwargs.get("mode", None), "MASKPACK")
    helper(
        f"MicroAPI::MaskPack<{mode}>({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)});"
    )


def handle_micro_maskunpack(inst, helper, expr_map) -> None:
    dst = _require_mask(inst.kwargs.get("dst", None), "MASKUNPACK")
    src = _require_mask(inst.kwargs.get("src", None), "MASKUNPACK")
    mode = _require_str(inst.kwargs.get("mode", None), "MASKUNPACK")
    helper(
        f"MicroAPI::MaskUnPack<{mode}>({value_to_cpp(dst, expr_map)}, {value_to_cpp(src, expr_map)});"
    )


MICRO_OP_HANDLERS: Dict[str, Handler] = {
    "micro_ub2reg": handle_micro_ub2reg,
    "micro_reg2ub": handle_micro_reg2ub,
    "micro_ub2regcont": handle_micro_ub2regcont,
    "micro_reg2ubcont": handle_micro_reg2ubcont,
    "micro_vexp": handle_micro_vexp,
    "micro_vabs": handle_micro_vabs,
    "micro_vrelu": handle_micro_vrelu,
    "micro_vsqrt": handle_micro_vsqrt,
    "micro_vln": handle_micro_vln,
    "micro_vlog": handle_micro_vlog,
    "micro_vlog2": handle_micro_vlog2,
    "micro_vlog10": handle_micro_vlog10,
    "micro_vneg": handle_micro_vneg,
    "micro_vnot": handle_micro_vnot,
    "micro_vcopy": handle_micro_vcopy,
    "micro_vmax": handle_micro_vmax,
    "micro_vmin": handle_micro_vmin,
    "micro_vadd": handle_micro_vadd,
    "micro_vsub": handle_micro_vsub,
    "micro_vmul": handle_micro_vmul,
    "micro_vdiv": handle_micro_vdiv,
    "micro_vand": handle_micro_vand,
    "micro_vor": handle_micro_vor,
    "micro_vxor": handle_micro_vxor,
    "micro_vprelu": handle_micro_vprelu,
    "micro_vmaxs": handle_micro_vmaxs,
    "micro_vmins": handle_micro_vmins,
    "micro_vadds": handle_micro_vadds,
    "micro_vmuls": handle_micro_vmuls,
    "micro_vlrelu": handle_micro_vlrelu,
    "micro_shiftls": handle_micro_shiftls,
    "micro_shiftrs": handle_micro_shiftrs,
    "micro_vaxpy": handle_micro_vaxpy,
    "micro_vcadd": handle_micro_vcadd,
    "micro_vcmax": handle_micro_vcmax,
    "micro_vcmin": handle_micro_vcmin,
    "micro_vcgadd": handle_micro_vcgadd,
    "micro_vcgmax": handle_micro_vcgmax,
    "micro_vcgmin": handle_micro_vcgmin,
    "micro_vcpadd": handle_micro_vcpadd,
    "micro_vdup": handle_micro_vdup,
    "micro_dinterleave": handle_micro_dinterleave,
    "micro_interleave": handle_micro_interleave,
    "micro_cast": handle_micro_cast,
    "micro_arange": handle_micro_arange,
    "micro_datacopygather": handle_micro_datacopygather,
    "micro_datacopyscatter": handle_micro_datacopyscatter,
    "micro_gather": handle_micro_gather,
    "micro_gathermask": handle_micro_gathermask,
    "micro_compare": handle_micro_compare,
    "micro_compares": handle_micro_compares,
    "micro_select": handle_micro_select,
    "micro_maskmov": handle_micro_maskmov,
    "micro_maskand": handle_micro_maskand,
    "micro_maskor": handle_micro_maskor,
    "micro_maskxor": handle_micro_maskxor,
    "micro_masknot": handle_micro_masknot,
    "micro_maskinterl": handle_micro_maskinterl,
    "micro_maskdeinterl": handle_micro_maskdeinterl,
    "micro_masksel": handle_micro_masksel,
    "micro_movemaskspr": handle_micro_movemaskspr,
    "micro_updatemask": handle_micro_updatemask,
    "micro_maskpack": handle_micro_maskpack,
    "micro_maskunpack": handle_micro_maskunpack,
}
