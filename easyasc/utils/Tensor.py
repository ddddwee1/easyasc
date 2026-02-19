from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union, cast
from .datatype import DataTypeValue
from .var import Var
from .positions import Position, PositionType
from .pipe import Pipe, PipeType
from .. import globvars
from .instruction import Instruction

if TYPE_CHECKING:
    from .vecop import VecOP
    from .mutex import CvMutex, VcMutex
    from .reg import Reg, RegList
    from .regop import RegOP


ShapeDim = Union[int, Var]
Shape2D = Sequence[ShapeDim]
ScalarIndex = Union[Var, int]
TensorIndex = Union[ScalarIndex, slice, Tuple[slice, ...]]
GMTensorDimIndex = Union[ScalarIndex, slice]
GMTensorIndex = Union[GMTensorDimIndex, Tuple[GMTensorDimIndex, ...]]


class Tensor:
    """Tensor class."""
    def __init__(
        self,
        dtype: DataTypeValue,
        shape: Shape2D,
        position: PositionType = Position.L1,
        name: str = "",
    ):
        """
        Initialize a tensor.

        Args:
            dtype: Data type, must be DataTypeValue.
            shape: Shape, list/tuple with length 2.
            position: Position type, defaults to Position.L1.
            name: Name, defaults to empty string.
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be list/tuple, got: {type(shape)}")
        
        if len(shape) != 2:
            raise ValueError(f"shape length must be 2, current length: {len(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        if not isinstance(position, PositionType):
            raise TypeError(f"position must be PositionType, got: {type(position)}")
        
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
        self.vec_copy_mode = ""

        target = globvars.active_micro if globvars.active_micro is not None else globvars.active_kernel
        if target is not None:
            target.instructions.append(
                Instruction("create_tensor", val=self, shape=list(self.shape))
            )

    def __repr__(self):
        return (
            f"Tensor(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"position={self.position!r}, idx={self.idx!r})"
        )

    def set_shape(self, shape: Shape2D) -> None:
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be list/tuple, got: {type(shape)}")
        if len(shape) != 2:
            raise ValueError(f"shape length must be 2, current length: {len(shape)}")
        self.shape = list(shape)
        self.span = list(shape)

    def T(self) -> "Tensor":
        if self.position is not Position.L1:
            raise ValueError(f"Tensor position must be L1, current position: {self.position}")
        # Create a view without emitting create_tensor.
        out = object.__new__(Tensor)
        out.__dict__ = self.__dict__.copy()
        out.is_transpose = True
        return out

    def __ilshift__(self, other: Union["GMTensor", "Tensor", "VecOP", "Reg", "RegList", "RegOP"]) -> "Tensor":
        from .vecop import VecOP
        from .reg import Reg, RegList
        from .regop import RegOP
        if isinstance(other, VecOP):
            if self.position is not Position.UB:
                raise ValueError(f"VecOP only supports UB position, current position: {self.position}")
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
            raise ValueError(f"Tensor position must be L1 or UB, current position: {self.position}")
        if isinstance(other, Tensor):
            if self.position is Position.L1 and other.position is Position.L0C:
                from ..stub_functions.cube import l0c_to_l1
                l0c_to_l1(self, other)
                return self
            if self.position not in (Position.L0A, Position.L0B):
                raise ValueError(f"Tensor position must be L0A or L0B, current position: {self.position}")
            new_shape = list(other.span) if hasattr(other, "span") else list(other.shape)
            self.set_shape(new_shape)
            if self.source_buf is not None:
                self.source_buf.shape = list(new_shape)
            from ..stub_functions.cube import l1_to_l0
            l1_to_l0(self, other)
            return self
        if isinstance(other, Reg):
            from ..stub_functions.micro.datamove import reg_to_ub, reg_to_ub_pack4, reg_to_ub_downsample
            if self.dtype == other.dtype:
                reg_to_ub(self, other)
            else:
                micro = globvars.active_micro
                if micro is None:
                    raise ValueError('Register related datamove must be called within vf')
                tmp = micro.get_reg(self.dtype)
                tmp <<= other.astype(self.dtype)
                if self.dtype.size==1 and other.dtype.size==4:
                    reg_to_ub_pack4(self, tmp)
                elif ((self.dtype.size==1 and other.dtype.size==2) or 
                      (self.dtype.size==2 and other.dtype.size==4)):
                    reg_to_ub_downsample(self, tmp)
                micro.release_reg(tmp)
            return self
        if isinstance(other, RegList):
            block = 256 // other.dtype.size
            for i in range(other.length):
                self[block * i] <<= other[i]
            return self
        if isinstance(other, RegOP):
            other.release_inputs()
            if other.opname in (
                "reg_to_ub",
                "reg_to_ub_continuous",
                "reg_to_ub_downsample",
                "reg_to_ub_pack4",
                "reg_to_ub_single",
                "reg_to_ub_scatter",
            ):
                other.emit(self)
                return self
            from ..stub_functions.micro.datamove import (
                reg_to_ub_downsample,
                reg_to_ub_pack4,
                reg_to_ub_single,
                reg_to_ub,
            )
            if other.opname == "reg_to_ub_downsample":
                src = other.inputs[0]
                if not isinstance(src, Reg):
                    raise TypeError(f"src must be Reg, got: {type(src)}")
                reg_to_ub_downsample(self, src, mask=other.mask)
                return self
            if other.opname == "reg_to_ub_pack4":
                src = other.inputs[0]
                if not isinstance(src, Reg):
                    raise TypeError(f"src must be Reg, got: {type(src)}")
                reg_to_ub_pack4(self, src, mask=other.mask)
                return self
            if other.opname == "reg_to_ub_single":
                src = other.inputs[0]
                if not isinstance(src, Reg):
                    raise TypeError(f"src must be Reg, got: {type(src)}")
                reg_to_ub_single(self, src, mask=other.mask)
                return self
            if other.opname == "vcopy":
                src = other.inputs[0]
                if not isinstance(src, Reg):
                    raise TypeError(f"src must be Reg, got: {type(src)}")
                reg_to_ub(self, src, mask=other.mask)
                return self
            tmp = other.run_regop()
            reg_to_ub(self, tmp)
            micro = globvars.active_micro
            if micro is not None and tmp.name.startswith("_tmp_reg_"):
                micro.release_reg(tmp)
            return self
        raise TypeError(f"other must be GMTensor/Tensor/VecOP/Reg/RegList/RegOP, got: {type(other)}")

    def _vecop(self, op: str, other: object = None) -> "VecOP":
        if self.position is not Position.UB:
            raise ValueError(f"VecOP only supports UB position, current position: {self.position}")
        from .vecop import VecOP
        return VecOP(op, self, other)  # type: ignore[arg-type]

    def _with_vec_copy_mode(self, mode: str) -> "Tensor":
        out = object.__new__(Tensor)
        out.__dict__ = self.__dict__.copy()
        out.vec_copy_mode = mode
        return out

    def downsample(self) -> "Tensor":
        return self._with_vec_copy_mode("downsample")

    def upsample(self) -> "Tensor":
        return self._with_vec_copy_mode("upsample")

    def unpack(self) -> "Tensor":
        return self._with_vec_copy_mode("unpack")

    def unpack4(self) -> "Tensor":
        return self._with_vec_copy_mode("unpack4")

    def brcb(self) -> "Tensor":
        return self._with_vec_copy_mode("brcb")

    def single(self) -> "Tensor":
        return self._with_vec_copy_mode("single")

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

    def __setitem__(self, *args, **kwargs):
        ...

    def __getitem__(self, index: TensorIndex) -> "Tensor":
        scalar_last_dim_offset: Optional[ScalarIndex] = None
        if isinstance(index, (Var, int)):
            scalar_last_dim_offset = index
            normalized_index: Tuple[slice, ...] = tuple(
                slice(None, None, None) for _ in self.shape
            )
        elif isinstance(index, slice):
            normalized_index = (index,)
        elif isinstance(index, tuple):
            normalized_index = index
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
        if len(normalized_index) != len(self.shape):
            raise TypeError(f"Tensor index dimensions must match shape, current length: {len(normalized_index)}")

        def _parse_dim(dim_index: slice, dim_size: ShapeDim, label: str) -> Tuple[ShapeDim, ShapeDim, int]:
            if not isinstance(dim_index, slice):
                raise TypeError(f"{label} index must be slice, got: {type(dim_index)}")

            start: ShapeDim = dim_index.start if dim_index.start is not None else 0
            stop: ShapeDim = dim_index.stop if dim_index.stop is not None else dim_size
            step = dim_index.step

            if not isinstance(start, (Var, int)):
                raise TypeError(f"{label} start must be Var or int, got: {type(start)}")
            if not isinstance(stop, (Var, int)):
                raise TypeError(f"{label} stop must be Var or int, got: {type(stop)}")
            if step is not None and step != 1:
                raise ValueError("slice step must be None or 1")

            span = cast(ShapeDim, stop - start)
            return start, span, 1

        offsets: List[ShapeDim] = []
        spans: List[ShapeDim] = []
        steps: List[int] = []
        for dim_idx, dim_index in enumerate(normalized_index):
            dim_size = self.span[dim_idx] if hasattr(self, "span") else self.shape[dim_idx]
            off, span, step = _parse_dim(dim_index, dim_size, f"dim{dim_idx}")
            offsets.append(off)
            spans.append(span)
            steps.append(step)
        if scalar_last_dim_offset is not None:
            offsets[-1] = scalar_last_dim_offset

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
        target = globvars.active_micro if globvars.active_micro is not None else globvars.active_kernel
        if target is not None:
            opname = "micro_slice_tensor" if globvars.active_micro is not None else "slice_tensor"
            target.instructions.append(
                Instruction(
                    opname,
                    src=self,
                    out=out,
                    offset=offsets,
                    span=spans,
                    step=steps,
                )
            )
        return out


class DBuff:
    """Data buffer class (double buffer composed of two Tensors)."""
    def __init__(
        self,
        dtype: DataTypeValue,
        shape: Shape2D,
        position: PositionType = Position.L1,
        name: str = "",
    ):
        """
        Initialize a data buffer.

        Args:
            dtype: Data type, must be DataTypeValue.
            shape: Shape, list/tuple with length 2.
            position: Position type, defaults to Position.L1.
            name: Name, defaults to empty string.
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be list/tuple, got: {type(shape)}")
        
        if len(shape) != 2:
            raise ValueError(f"shape length must be 2, current length: {len(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        if not isinstance(position, PositionType):
            raise TypeError(f"position must be PositionType, got: {type(position)}")
        
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
                Instruction("create_dbuf", val=self, shape=list(self.shape))
            )

    def __repr__(self):
        return (
            f"DBuff(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"position={self.position!r}, idx={self.idx!r})"
        )

    # DBuff represents a double buffer of two Tensors; indexing returns one Tensor view.
    def __getitem__(self, index: ScalarIndex) -> "Tensor":
        if not isinstance(index, (Var, int)):
            raise TypeError(f"index must be Var or int, got: {type(index)}")

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

    def __setitem__(self, index: ScalarIndex, value: "Tensor") -> None:
        if not isinstance(index, (Var, int)):
            raise TypeError(f"index must be Var or int, got: {type(index)}")
        if not isinstance(value, Tensor):
            raise TypeError(f"value must be Tensor, got: {type(value)}")


class GMTensor:
    """Global-memory tensor class."""
    def __init__(self, dtype: DataTypeValue, shape: Shape2D, name: str = ""):
        """
        Initialize a global-memory tensor.

        Args:
            dtype: Data type, must be DataTypeValue.
            shape: Shape, list/tuple with length 2.
            name: Name, defaults to empty string.
        """
        if not isinstance(dtype, DataTypeValue):
            raise TypeError(f"dtype must be DataTypeValue, got: {type(dtype)}")
        
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be list/tuple, got: {type(shape)}")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")
        
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
        self._mutex: Optional[Union["CvMutex", "VcMutex"]] = None
        
        if globvars.active_kernel is not None:
            globvars.active_kernel.instructions.append(
                Instruction("create_gm_tensor", val=self)
            )

    def __repr__(self):
        return (
            f"GMTensor(name={self.name!r}, dtype={self.dtype!r}, shape={self.shape!r}, "
            f"idx={self.idx!r})"
        )

    def bind_cv_mutex(
        self,
        flag_id: int,
        depth: int = 2,
        src_start_pipe: PipeType = Pipe.S,
        dst_start_pipe: PipeType = Pipe.S,
        src_end_pipe: PipeType = Pipe.FIX,
        dst_end_pipe: Optional[PipeType] = None,
    ) -> "CvMutex":
        from .mutex import CvMutex

        mutex = CvMutex(
            flag_id,
            depth,
            src_start_pipe=src_start_pipe,
            dst_start_pipe=dst_start_pipe,
            src_end_pipe=src_end_pipe,
            dst_end_pipe=dst_end_pipe,
        )
        self._mutex = mutex
        return mutex

    def bind_vc_mutex(
        self,
        flag_id: int,
        depth: int = 2,
        src_start_pipe: PipeType = Pipe.S,
        dst_start_pipe: PipeType = Pipe.S,
        src_end_pipe: PipeType = Pipe.MTE3,
        dst_end_pipe: PipeType = Pipe.FIX,
    ) -> "VcMutex":
        from .mutex import VcMutex

        mutex = VcMutex(
            flag_id,
            depth,
            src_start_pipe=src_start_pipe,
            dst_start_pipe=dst_start_pipe,
            src_end_pipe=src_end_pipe,
            dst_end_pipe=dst_end_pipe,
        )
        self._mutex = mutex
        return mutex

    def lock(self) -> None:
        if self._mutex is None:
            raise ValueError("GMTensor has not bound a mutex yet")
        from .mutex import CvMutex, VcMutex
        if not isinstance(self._mutex, (CvMutex, VcMutex)):
            raise TypeError(f"GMTensor has invalid mutex type: {type(self._mutex)}")
        self._mutex.lock()

    def ready(self) -> None:
        if self._mutex is None:
            raise ValueError("GMTensor has not bound a mutex yet")
        from .mutex import CvMutex, VcMutex
        if not isinstance(self._mutex, (CvMutex, VcMutex)):
            raise TypeError(f"GMTensor has invalid mutex type: {type(self._mutex)}")
        self._mutex.ready()

    def wait(self) -> None:
        if self._mutex is None:
            raise ValueError("GMTensor has not bound a mutex yet")
        from .mutex import CvMutex, VcMutex
        if not isinstance(self._mutex, (CvMutex, VcMutex)):
            raise TypeError(f"GMTensor has invalid mutex type: {type(self._mutex)}")
        self._mutex.wait()

    def free(self) -> None:
        if self._mutex is None:
            raise ValueError("GMTensor has not bound a mutex yet")
        from .mutex import CvMutex, VcMutex
        if not isinstance(self._mutex, (CvMutex, VcMutex)):
            raise TypeError(f"GMTensor has invalid mutex type: {type(self._mutex)}")
        self._mutex.free()

    def __ilshift__(self, other: "Tensor") -> "GMTensor":
        if isinstance(other, Tensor):
            if other.position is Position.UB:
                from ..stub_functions.vec.datamove import ub_to_gm_pad
                ub_to_gm_pad(self, other)
                return self
            if other.position is Position.L0C:
                from ..stub_functions.cube import l0c_to_gm_nz2nd
                l0c_to_gm_nz2nd(self, other)
                return self
            raise ValueError(f"Tensor position must be L0C or UB, current position: {other.position}")
        raise TypeError(f"other must be Tensor, got: {type(other)}")

    def __getitem__(self, index: GMTensorIndex) -> "GMTensor":
        if isinstance(index, tuple):
            normalized_index = index
        else:
            normalized_index = (index,)
        if len(normalized_index) != len(self.shape):
            raise TypeError(f"GMTensor index dimensions must match shape, current length: {len(normalized_index)}")

        def _parse_dim(dim_index: GMTensorDimIndex, dim_size: ShapeDim, label: str) -> Tuple[ShapeDim, ShapeDim, int]:
            if isinstance(dim_index, (Var, int)):
                return dim_index, 1, 1
            if not isinstance(dim_index, slice):
                raise TypeError(f"{label} index must be Var, int, or slice, got: {type(dim_index)}")

            start: ShapeDim = dim_index.start if dim_index.start is not None else 0
            stop: ShapeDim = dim_index.stop if dim_index.stop is not None else dim_size
            step = dim_index.step

            if not isinstance(start, (Var, int)):
                raise TypeError(f"{label} start must be Var or int, got: {type(start)}")
            if not isinstance(stop, (Var, int)):
                raise TypeError(f"{label} stop must be Var or int, got: {type(stop)}")
            if step is not None and step != 1:
                raise ValueError("slice step must be None or 1")

            span = cast(ShapeDim, stop - start)
            return start, span, 1

        slice_count = 0
        offsets: List[ShapeDim] = []
        spans: List[ShapeDim] = []
        steps: List[int] = []
        slice_mask: List[bool] = []
        for dim_idx, dim_index in enumerate(normalized_index):
            dim_size = self.span[dim_idx] if hasattr(self, "span") else self.shape[dim_idx]
            is_slice = isinstance(dim_index, slice)
            if is_slice:
                slice_count += 1
                if slice_count > 2:
                    raise TypeError("GMTensor cannot have slices on more than 2 dimensions")
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
        out._mutex = getattr(self, "_mutex", None)
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

    def __setitem__(self, index: GMTensorIndex, value: object) -> None:
        if isinstance(value, Tensor):
            if value.position is not Position.L0C:
                raise ValueError(f"Tensor position must be L0C, current position: {value.position}")
            dst = self.__getitem__(index)
            from ..stub_functions.cube import l0c_to_gm_nz2nd
            l0c_to_gm_nz2nd(dst, value)
            return
        if isinstance(value, GMTensor):
            return
        raise TypeError(f"value must be Tensor or GMTensor, got: {type(value)}")
