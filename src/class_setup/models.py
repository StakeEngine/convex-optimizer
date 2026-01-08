from dataclasses import dataclass


@dataclass
class LogNormalParams:
    mode: float = 1.0
    std: float = 1.0
    mean: float = 1.0
    scale: float = 1.0
    linear_offset: float = 0.0

    the_exp: float = 0.0

    target_mode: float = 0.0
    target_std: float = 0.0
    target_mean: float = 0.0

    xmin: float = 0.0
    xmax: float = 0.0


@dataclass
class GaussianParams:
    mean: float = 1.0
    std: float = 1.0
    scale: float = 1.0
    linear_offset: float = 0.0

    xmin: float = 0.0
    xmax: float = 0.0


@dataclass
class ExponentialParams:
    scale: float = 1.0
    power: float = 1.0
    linear_offset: float = 0.0

    xmin: float = 0.0
    xmax: float = 0.0
