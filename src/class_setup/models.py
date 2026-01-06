from dataclasses import dataclass


@dataclass
class LogNormalParams:
    mode: float = 1.0
    std: float = 1.0
    mean: float = 1.0
    scale: float = 1.0

    the_exp: float = 0.0

    target_mode: float = 0.0
    target_std: float = 0.0
    target_mean: float = 0.0

    xlim: float = 0.0
    ylim: float = 0.0


@dataclass
class GaussianParams:
    mean: float = 1.0
    std: float = 1.0
    scale: float = 1.0

    xlim: float = 0.0
    ylim: float = 0.0


@dataclass
class ExponentialParams:
    scale: float = 1.0
    power: float = 1.0

    xlim: float = 0.0
    ylim: float = 0.0
