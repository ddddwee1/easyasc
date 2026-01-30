from typing import Union
from .datatype import DataTypeValue
from .var import Var
from .positions import Position, PositionType
from .. import globvars
from .instruction import Instruction


class Tensor:
    """张量类"""
    def __init__(
        self,
        dtype: DataTypeValue,
        shape: Union[list, tuple],
        position: PositionType = Position.L1,
        name: str = "",
    ):
        """
        初始化张量
        
        Args:
            dtype: 数据类型，必须是DataTypeValue类型
            shape: 形状，长度为2的list或tuple
            position: 位置类型，默认为Position.L1
            name: 名称，默认为空字符串
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape必须是list或tuple类型，当前类型: {type(shape)}")
        
        if len(shape) != 2:
            raise ValueError(f"shape长度必须为2，当前长度: {len(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if not isinstance(position, PositionType):
            raise TypeError(f"position必须是PositionType类型，当前类型: {type(position)}")
        
        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        if name == "":
            name = f"_tmp_tensor_{idx}"

        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.position = position
        self.idx = idx
        self.offset = [0 for _ in shape]
        self.span = list(shape)
        self.step = [1 for _ in shape]
        self.source_buf: Union[None, 'DBuff'] = None
        self.source_index: Union[None, 'Var', int] = None
        self.is_transpose = False

        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("create_tensor", val=self)
            )

    def __repr__(self):
        return (
            f"Tensor(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"position={self.position!r}, idx={self.idx!r})"
        )

    def set_shape(self, shape: Union[list, tuple]) -> None:
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape必须是list或tuple类型，当前类型: {type(shape)}")
        if len(shape) != 2:
            raise ValueError(f"shape长度必须为2，当前长度: {len(shape)}")
        self.shape = list(shape)
        self.span = list(shape)

    def T(self) -> "Tensor":
        if self.position is not Position.L1:
            raise ValueError(f"Tensor位置必须为L1，当前位置: {self.position}")
        # Create a view without emitting create_tensor.
        out = object.__new__(Tensor)
        out.__dict__ = self.__dict__.copy()
        out.is_transpose = True
        return out

    def __ilshift__(self, other: Union["GMTensor", "Tensor", "VecOP"]) -> "Tensor":
        from .vecop import VecOP
        if isinstance(other, VecOP):
            if self.position is not Position.UB:
                raise ValueError(f"VecOP仅支持UB位置，当前位置: {self.position}")
            other.emit(self)
            return self
        if isinstance(other, GMTensor):
            if self.position is Position.UB:
                from ..stub_functions.vec.datamove import gm_to_ub_pad
                gm_to_ub_pad(self, other)
                return self
            if self.position is Position.L1:
                from ..stub_functions.cube import gm_to_l1_nd2nz
                gm_to_l1_nd2nz(self, other)
                return self
            raise ValueError(f"Tensor位置必须为L1或UB，当前位置: {self.position}")
        if isinstance(other, Tensor):
            if self.position not in (Position.L0A, Position.L0B):
                raise ValueError(f"Tensor位置必须为L0A或L0B，当前位置: {self.position}")
            new_shape = list(other.span) if hasattr(other, "span") else list(other.shape)
            self.set_shape(new_shape)
            if self.source_buf is not None:
                self.source_buf.shape = list(new_shape)
            from ..stub_functions.cube import l1_to_l0
            l1_to_l0(self, other)
            return self
        raise TypeError(f"other必须是GMTensor或Tensor类型，当前类型: {type(other)}")

    def _vecop(self, op: str, other: object = None) -> "VecOP":
        if self.position is not Position.UB:
            raise ValueError(f"VecOP仅支持UB位置，当前位置: {self.position}")
        from .vecop import VecOP
        return VecOP(op, self, other)  # type: ignore[arg-type]

    def __add__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("add", other)
        if isinstance(other, (int, float, Var)):
            return self._vecop("adds", other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float, Var)):
            return self._vecop("adds", other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("sub", other)
        if isinstance(other, (int, float)):
            return self._vecop("adds", -other)
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("mul", other)
        if isinstance(other, (int, float, Var)):
            return self._vecop("muls", other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, Var)):
            return self._vecop("muls", other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("div", other)
        if isinstance(other, (int, float)) and other != 0:
            return self._vecop("muls", 1.0 / other)
        return NotImplemented

    def __rtruediv__(self, other):
        return NotImplemented

    def exp(self):
        return self._vecop("exp")

    def ln(self):
        return self._vecop("ln")

    def abs(self):
        return self._vecop("abs")

    def rec(self):
        return self._vecop("rec")

    def sqrt(self):
        return self._vecop("sqrt")

    def rsqrt(self):
        return self._vecop("rsqrt")

    def vnot(self):
        return self._vecop("vnot")

    def relu(self):
        return self._vecop("relu")

    def cmax(self):
        return self._vecop("cmax")

    def cgmax(self):
        return self._vecop("cgmax")

    def cmin(self):
        return self._vecop("cmin")

    def cgmin(self):
        return self._vecop("cgmin")

    def cadd(self):
        return self._vecop("cadd")

    def cgadd(self):
        return self._vecop("cgadd")

    def cpadd(self):
        return self._vecop("cpadd")

    def cast(self):
        return self._vecop("cast")

    def maximum(self, other):
        if isinstance(other, Tensor):
            return self._vecop("max", other)
        if isinstance(other, (int, float, Var)):
            return self._vecop("maxs", other)
        return NotImplemented

    def minimum(self, other):
        if isinstance(other, Tensor):
            return self._vecop("min", other)
        if isinstance(other, (int, float, Var)):
            return self._vecop("mins", other)
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("and", other)
        return NotImplemented

    def __rand__(self, other):
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, Tensor):
            return self._vecop("or", other)
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __getitem__(self, index: Union[slice, tuple]) -> "Tensor":
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise TypeError(f"Tensor索引维度必须与shape一致，当前长度: {len(index)}")

        def _parse_dim(dim_index: slice, dim_size, label: str):
            if not isinstance(dim_index, slice):
                raise TypeError(f"{label}索引必须是slice类型，当前类型: {type(dim_index)}")

            start = dim_index.start if dim_index.start is not None else 0
            stop = dim_index.stop if dim_index.stop is not None else dim_size
            step = dim_index.step

            if not isinstance(start, (Var, int)):
                raise TypeError(f"{label} start必须是Var或int类型，当前类型: {type(start)}")
            if not isinstance(stop, (Var, int)):
                raise TypeError(f"{label} stop必须是Var或int类型，当前类型: {type(stop)}")
            if step is not None and step != 1:
                raise ValueError("slice step must be None or 1")

            span = stop - start
            return start, span, 1

        offsets = []
        spans = []
        steps = []
        for dim_idx, dim_index in enumerate(index):
            dim_size = self.span[dim_idx] if hasattr(self, "span") else self.shape[dim_idx]
            off, span, step = _parse_dim(dim_index, dim_size, f"dim{dim_idx}")
            offsets.append(off)
            spans.append(span)
            steps.append(step)

        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        name = f"_tmp_tensor_{idx}"
        out = object.__new__(Tensor)
        out.dtype = self.dtype
        out.shape = list(self.shape)
        out.name = name
        out.position = self.position
        out.idx = idx
        out.offset = offsets
        out.span = spans
        out.step = steps
        out.source_buf = self.source_buf
        out.source_index = self.source_index
        out.is_transpose = self.is_transpose
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction(
                    "slice_tensor",
                    src=self,
                    out=out,
                    offset=offsets,
                    span=spans,
                    step=steps,
                )
            )
        return out


class DBuff:
    """数据缓冲区类（double buffer，由两个Tensor合并而成）"""
    def __init__(
        self,
        dtype: DataTypeValue,
        shape: Union[list, tuple],
        position: PositionType = Position.L1,
        name: str = "",
    ):
        """
        初始化数据缓冲区
        
        Args:
            dtype: 数据类型，必须是DataTypeValue类型
            shape: 形状，长度为2的list或tuple
            position: 位置类型，默认为Position.L1
            name: 名称，默认为空字符串
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape必须是list或tuple类型，当前类型: {type(shape)}")
        
        if len(shape) != 2:
            raise ValueError(f"shape长度必须为2，当前长度: {len(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")

        if not isinstance(position, PositionType):
            raise TypeError(f"position必须是PositionType类型，当前类型: {type(position)}")
        
        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        if name == "":
            name = f"_tmp_dbuf_{idx}"

        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.position = position
        self.idx = idx

        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("create_dbuf", val=self)
            )

    def __repr__(self):
        return (
            f"DBuff(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"position={self.position!r}, idx={self.idx!r})"
        )

    # DBuff代表两个Tensor的double buffer，索引返回其中一个Tensor视图
    def __getitem__(self, index: Union[Var, int]) -> "Tensor":
        if not isinstance(index, (Var, int)):
            raise TypeError(f"index必须是Var或int类型，当前类型: {type(index)}")

        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        name = f"_tmp_tensor_{idx}"
        out = object.__new__(Tensor)
        out.dtype = self.dtype
        out.shape = list(self.shape)
        out.name = name
        out.position = self.position
        out.idx = idx
        out.offset = [0 for _ in out.shape]
        out.span = list(out.shape)
        out.step = [1 for _ in out.shape]
        out.source_buf = None
        out.source_index = None
        out.is_transpose = False
        out.source_buf = self
        out.source_index = index
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("get_buf", buf=self, index=index, out=out)
            )
        return out

    def __setitem__(self, index: Union[Var, int], value: "Tensor") -> None:
        if not isinstance(index, (Var, int)):
            raise TypeError(f"index必须是Var或int类型，当前类型: {type(index)}")
        if not isinstance(value, Tensor):
            raise TypeError(f"value必须是Tensor类型，当前类型: {type(value)}")


class GMTensor:
    """全局内存张量类"""
    def __init__(self, dtype: DataTypeValue, shape: Union[list, tuple], name: str = ""):
        """
        初始化全局内存张量
        
        Args:
            dtype: 数据类型，必须是DataTypeValue类型
            shape: 形状，长度为2的list或tuple
            name: 名称，默认为空字符串
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype必须是DataTypeValue类型，当前类型: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape必须是list或tuple类型，当前类型: {type(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name必须是str类型，当前类型: {type(name)}")
        
        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        if name == "":
            name = f"_tmp_gmtensor_{idx}"
        
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.idx = idx
        self.offset = [0 for _ in shape]
        self.span = list(shape)
        self.step = [1 for _ in shape]
        self.slice_mask = [False for _ in shape]
        
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("create_gm_tensor", val=self)
            )

    def __repr__(self):
        return (
            f"GMTensor(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"idx={self.idx!r})"
        )

    def __ilshift__(self, other: "Tensor") -> "GMTensor":
        if isinstance(other, Tensor):
            if other.position is Position.UB:
                from ..stub_functions.vec.datamove import ub_to_gm_pad
                ub_to_gm_pad(self, other)
                return self
            if other.position is Position.L0C:
                from ..stub_functions.cube import l0c_to_gm_nd2nz
                l0c_to_gm_nd2nz(self, other)
                return self
            raise ValueError(f"Tensor位置必须为L0C或UB，当前位置: {other.position}")
        raise TypeError(f"other必须是Tensor类型，当前类型: {type(other)}")

    def __getitem__(self, index: Union[Var, int, slice, tuple]) -> "GMTensor":
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape):
            raise TypeError(f"GMTensor索引维度必须与shape一致，当前长度: {len(index)}")

        def _parse_dim(dim_index: Union[Var, int, slice], dim_size, label: str):
            if isinstance(dim_index, (Var, int)):
                return dim_index, 1, 1
            if not isinstance(dim_index, slice):
                raise TypeError(f"{label}索引必须是Var、int或slice类型，当前类型: {type(dim_index)}")

            start = dim_index.start if dim_index.start is not None else 0
            stop = dim_index.stop if dim_index.stop is not None else dim_size
            step = dim_index.step

            if not isinstance(start, (Var, int)):
                raise TypeError(f"{label} start必须是Var或int类型，当前类型: {type(start)}")
            if not isinstance(stop, (Var, int)):
                raise TypeError(f"{label} stop必须是Var或int类型，当前类型: {type(stop)}")
            if step is not None and step != 1:
                raise ValueError("slice step must be None or 1")

            span = stop - start
            return start, span, 1

        slice_count = 0
        offsets = []
        spans = []
        steps = []
        slice_mask = []
        for dim_idx, dim_index in enumerate(index):
            dim_size = self.span[dim_idx] if hasattr(self, "span") else self.shape[dim_idx]
            is_slice = isinstance(dim_index, slice)
            if is_slice:
                slice_count += 1
                if slice_count > 2:
                    raise TypeError("GMTensor不能有超过2维的slice")
            off, span, step = _parse_dim(dim_index, dim_size, f"dim{dim_idx}")
            offsets.append(off)
            spans.append(span)
            steps.append(step)
            slice_mask.append(is_slice)

        idx = globvars.tmp_idx
        globvars.tmp_idx += 1
        name = f"_tmp_gmtensor_{idx}"
        out = object.__new__(GMTensor)
        out.dtype = self.dtype
        out.shape = list(self.shape)
        out.name = name
        out.idx = idx
        out.offset = offsets
        out.span = spans
        out.step = steps
        out.slice_mask = slice_mask
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction(
                    "slice_gm_tensor",
                    src=self,
                    out=out,
                    offset=offsets,
                    span=spans,
                    step=steps,
                    slice_mask=slice_mask,
                )
            )
        return out

    def __setitem__(self, index: Union[Var, int, slice, tuple], value: object) -> None:
        if isinstance(value, Tensor):
            if value.position is not Position.L0C:
                raise ValueError(f"Tensor位置必须为L0C，当前位置: {value.position}")
            dst = self.__getitem__(index)
            from ..stub_functions.cube import l0c_to_gm_nd2nz
            l0c_to_gm_nd2nz(dst, value)
            return
        if isinstance(value, GMTensor):
            return
        raise TypeError(f"value必须是Tensor或GMTensor类型，当前类型: {type(value)}")
