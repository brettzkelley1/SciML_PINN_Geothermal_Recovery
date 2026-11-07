from dataclasses import dataclass, field

IN_TO_M = 0.0254
H_TO_S = 3600.0

@dataclass(frozen=True)
class RockProps:
    kappa: float = 3.00
    rho: float = 2500.0
    cp: float = 900.0
    @property
    def D(self) -> float:
        return self.kappa / (self.rho * self.cp)

@dataclass(frozen=True)
class Domain:
    r_w: float = 7 * IN_TO_M
    r_max: float = 6.5

@dataclass(frozen=True)
class WellCase:
    name: str
    t_c_h: float
    sft: float
    t0_bht: float
    t_max_h: float                  # per-well window (paper Figs 6-7 axes)
    rock: RockProps = field(default_factory=RockProps)
    domain: Domain = field(default_factory=Domain)

@dataclass(frozen=True)
class ReplicationOptions:
    """Defaults = paper mode. Flip these only in the improved copy."""
    nondimensionalize: bool = False   # paper appears to use raw units
    ic_loss: bool = False             # paper has no explicit IC term
    learn_D: bool = False             # variant A (fixed); True = variant B
    time_unit: str = "s"              # "s" or "h"; test both against Fig 3 (loss ~1.3e9)

@dataclass(frozen=True)
class PINNConfig:
    hidden_layers: int = 8
    width: int = 20
    activation: str = "tanh"
    lr: float = 1e-3
    epochs: int = 15_000
    seed: int = 0

