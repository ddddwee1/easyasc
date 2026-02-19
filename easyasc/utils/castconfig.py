from .. import globvars
from .roundmode import RoundMode, RoundModeType


class RegLayoutValue:
    """Register layout value class representing a concrete layout."""
    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")
        self.name = name

    def __repr__(self) -> str:
        return f"RegLayoutValue('{self.name}')"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, RegLayoutValue):
            raise TypeError(f"Cannot compare RegLayoutValue with {type(other)}")
        return self.name == other.name


class RegLayout:
    """Register layout enum-like class."""
    ZERO = RegLayoutValue("ZERO")
    ONE = RegLayoutValue("ONE")
    TWO = RegLayoutValue("TWO")
    THREE = RegLayoutValue("THREE")


class CastConfig:
    def __init__(
        self,
        round_mode: RoundModeType = RoundMode.AWAY_FROM_ZERO,
        reg_layout: RegLayoutValue = RegLayout.ZERO,
        saturate: bool = False,
        name: str = "",
    ) -> None:
        if not isinstance(round_mode, RoundModeType):
            raise TypeError(f"round_mode must be RoundModeType, got: {type(round_mode)}")
        if not isinstance(reg_layout, RegLayoutValue):
            raise TypeError(f"reg_layout must be RegLayoutValue, got: {type(reg_layout)}")
        if not isinstance(saturate, bool):
            raise TypeError(f"saturate must be bool, got: {type(saturate)}")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got: {type(name)}")

        self.reg_layout = reg_layout
        self.round_mode = round_mode
        self.saturate = saturate
        self.name = name

        micro = globvars.active_micro
        if micro is not None:
            cast_cfg_list = getattr(micro, "cast_cfg_list", None)
            if cast_cfg_list is None:
                cast_cfg_list = []
                setattr(micro, "cast_cfg_list", cast_cfg_list)
            if self not in cast_cfg_list:
                prefix = getattr(micro, "name", micro.__class__.__name__)
                self.name = f"{prefix}_{self.name}"
                cast_cfg_list.append(self)

    def __repr__(self) -> str:
        return (
            "CastConfig("
            f"reg_layout={self.reg_layout!r}, "
            f"round_mode={self.round_mode!r}, "
            f"saturate={self.saturate!r}, "
            f"name={self.name!r})"
        )

    def __str__(self) -> str:
        return self.name
