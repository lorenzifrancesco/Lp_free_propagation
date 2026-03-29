from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib

try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import BaseModel, Field, root_validator, validator

    PYDANTIC_V2 = False
    ConfigDict = None
    field_validator = None
    model_validator = None


def _parse_complex(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
        return complex(value, 0.0)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(value[0], value[1])
    if isinstance(value, str):
        return complex(value.replace(" ", ""))
    raise ValueError(f"Unsupported complex format: {value!r}")


class FiberConfig(BaseModel):
    v_number: float = Field(..., gt=0)
    n1: float = Field(..., gt=0)
    length: float = Field(..., gt=0)
    radius_units: float = Field(1.0, gt=0)
    radius_m: Optional[float] = Field(default=None, gt=0)

    if PYDANTIC_V2:
        @field_validator("radius_m", mode="before")
        def _coerce_radius_m(cls, value: Any) -> Optional[float]:
            if value in (None, "", "null", "none", "None", 0):
                return 30e-6
            return value
    else:
        @validator("radius_m", pre=True)
        def _coerce_radius_m(cls, value: Any) -> Optional[float]:
            if value in (None, "", "null", "none", "None", 0):
                return 30e-6
            return value


class BeamConfig(BaseModel):
    wavelength: float = Field(..., gt=0)
    dist_to_waist: float = Field(0.0, ge=0)
    w0_x: float = Field(..., gt=0)
    w0_y: float = Field(..., gt=0)
    x0: float = 0.0
    y0: float = 0.0
    roll_angle: float = 0.0
    pitch_angle: float = 0.0
    yaw_angle: float = 0.0
    polarization_angle: float = 0.0


class DomainConfig(BaseModel):
    axis_size: float = Field(..., gt=0)
    r_origin_factor: float = Field(1.3, gt=0)


class GeneratorConfig(BaseModel):
    modes_to_test: List[Tuple[int, int]] = Field(default_factory=list)
    use_custom_coeffs: bool = False
    dist_from_fiber: float = 0.0

    if PYDANTIC_V2:
        @field_validator("modes_to_test")
        def _validate_modes(cls, value: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            for l, m in value:
                if l < 0 or m < 1:
                    raise ValueError("Mode indices must satisfy l >= 0 and m >= 1.")
            return value
    else:
        @validator("modes_to_test")
        def _validate_modes(cls, value: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            for l, m in value:
                if l < 0 or m < 1:
                    raise ValueError("Mode indices must satisfy l >= 0 and m >= 1.")
            return value


class GridGeneratorConfig(BaseModel):
    grid_size: int = Field(..., gt=0)


class GridPropagationConfig(BaseModel):
    dx: float = Field(..., gt=0)
    oversampling_x: float = Field(..., gt=0)
    oversampling_z: float = Field(..., gt=0)
    rz_factor: float = Field(..., gt=0)
    k_max_factor: float = Field(..., gt=0)
    min_point_per_period: int = Field(..., gt=0)


class GridConfig(BaseModel):
    generator: GridGeneratorConfig
    propagation: GridPropagationConfig


class PropagationConfig(BaseModel):
    dist_from_fiber_start: float = Field(..., ge=0)
    dist_from_fiber_stop: float = Field(..., gt=0)
    dist_from_fiber_step: float = Field(..., gt=0)

    if PYDANTIC_V2:
        @model_validator(mode="after")
        def _validate_range(self) -> "PropagationConfig":
            if self.dist_from_fiber_stop <= self.dist_from_fiber_start:
                raise ValueError(
                    "dist_from_fiber_stop must be greater than dist_from_fiber_start."
                )
            return self
    else:
        @root_validator
        def _validate_range(cls, values: dict) -> dict:
            start = values.get("dist_from_fiber_start")
            stop = values.get("dist_from_fiber_stop")
            step = values.get("dist_from_fiber_step")
            if start is not None and stop is not None and step is not None:
                if stop <= start:
                    raise ValueError(
                        "dist_from_fiber_stop must be greater than dist_from_fiber_start."
                    )
            return values

    def dist_from_fiber_array(self) -> np.ndarray:
        return np.arange(
            self.dist_from_fiber_start,
            self.dist_from_fiber_stop,
            self.dist_from_fiber_step,
        )


class RuntimeConfig(BaseModel):
    supervision_mode: bool = False
    n_threads: int = Field(..., gt=0)
    recompute: bool = False


class VisualizationConfig(BaseModel):
    cmap: str = "gnuplot2"
    cmap_levels: int = Field(20, gt=0)


class ModeCoeff(BaseModel):
    l: int = Field(..., ge=0)
    m: int = Field(..., ge=1)
    x_p_phi: complex = 0 + 0j
    y_p_phi: complex = 0 + 0j
    x_m_phi: complex = 0 + 0j
    y_m_phi: complex = 0 + 0j

    if PYDANTIC_V2:
        @field_validator("x_p_phi", "y_p_phi", "x_m_phi", "y_m_phi", mode="before")
        def _coerce_complex(cls, value: Any) -> complex:
            return _parse_complex(value)

        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        @validator("x_p_phi", "y_p_phi", "x_m_phi", "y_m_phi", pre=True)
        def _coerce_complex(cls, value: Any) -> complex:
            return _parse_complex(value)

        class Config:
            arbitrary_types_allowed = True


class AppConfig(BaseModel):
    fiber: FiberConfig
    beam: BeamConfig
    domain: DomainConfig
    generator: GeneratorConfig
    grid: GridConfig
    propagation: PropagationConfig
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    custom_coeffs: List[ModeCoeff] = Field(default_factory=list)

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


def load_config(path: Union[Path, str] = "input/config.toml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if PYDANTIC_V2:
        return AppConfig.model_validate(data)
    return AppConfig.parse_obj(data)
