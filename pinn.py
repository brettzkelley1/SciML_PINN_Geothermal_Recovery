import math
import torch
import torch.nn as nn
from config import PINNConfig, RockProps, H_TO_S

TIME_UNITS = ("s", "h", "h_raw")   # h_raw = t in hours but D left in m^2/s (units mismatch)


class PINN(nn.Module):
    """MLP T(r, t). r in metres; t in `time_unit`; T in deg C (raw, paper mode).

    NOTE: the constructor calls torch.manual_seed(cfg.seed) so that initialisation is
    reproducible. Use your own torch.Generator for collocation sampling."""

    def __init__(self, cfg: PINNConfig | None = None, D0: float | None = None,
                 time_unit: str = "s", learn_D: bool = False,
                 in_scale=(1.0, 1.0), out_scale: float = 1.0, out_shift: float = 0.0,
                 init: str = "xavier", dtype=torch.float64):
        super().__init__()
        cfg = PINNConfig() if cfg is None else cfg
        assert cfg.activation == "tanh", "paper uses tanh"
        assert time_unit in TIME_UNITS
        D0 = RockProps().D if D0 is None else D0
        self.cfg, self.time_unit = cfg, time_unit

        torch.manual_seed(cfg.seed)
        dims = [2] + [cfg.width] * cfg.hidden_layers + [1]
        self.layers = nn.ModuleList(
            nn.Linear(a, b, dtype=dtype) for a, b in zip(dims[:-1], dims[1:]))
        if init == "xavier":
            for l in self.layers:
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)
        elif init != "torch_default":
            raise ValueError(init)

        # optional fixed input/output affine maps (identity = paper mode)
        self.register_buffer("in_scale", torch.tensor(in_scale, dtype=dtype))
        self.register_buffer("out_aff", torch.tensor([out_scale, out_shift], dtype=dtype))

        # diffusivity stored as log D: positive by construction, trainable if learn_D
        logD = torch.tensor(math.log(D0), dtype=dtype)
        if learn_D:
            self.log_D = nn.Parameter(logD)
        else:
            self.register_buffer("log_D", logD)

    # ---- network ----
    def forward(self, r, t):
        """r, t: tensors of shape (N, 1). Returns T of shape (N, 1)."""
        x = torch.cat([r, t], dim=1) / self.in_scale
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x) * self.out_aff[0] + self.out_aff[1]

    # ---- diffusivity and units ----
    @property
    def D(self):
        """Diffusivity, m^2/s."""
        return torch.exp(self.log_D)

    @property
    def D_t(self):
        """Diffusivity in m^2 per (input time unit), as used inside the residual."""
        if self.time_unit == "h":
            return self.D * H_TO_S
        return self.D              # "s" (correct) or "h_raw" (deliberately unconverted)

    def t_from_hours(self, t_h):
        return t_h * H_TO_S if self.time_unit == "s" else t_h

    def residual(self, r, t, return_terms: bool = False):
        return pde_residual(self, r, t, self.D_t, return_terms)


def pde_residual(field, r, t, D_t, return_terms: bool = False):
    """Paper Eq. 5 integrand: (1/D) T_t - T_rr - T_r / r, via autograd.
    `field(r, t)` is any callable returning T with shape (N, 1)."""
    r = r.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)
    T = field(r, t)
    T_t, T_r = torch.autograd.grad(T, (t, r), grad_outputs=torch.ones_like(T),
                                   create_graph=True)
    T_rr = torch.autograd.grad(T_r, r, grad_outputs=torch.ones_like(T_r),
                               create_graph=True)[0]
    a, b, c = T_t / D_t, T_rr, T_r / r
    res = a - b - c
    return (res, a, b, c) if return_terms else res

