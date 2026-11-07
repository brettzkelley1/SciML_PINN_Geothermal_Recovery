import math
from dataclasses import dataclass
import torch
from config import WellCase, H_TO_S
from data import WellData
from ic import gaussian_ic

DT = torch.float64


@dataclass(frozen=True)
class LossWeights:
    m: float = 1.0       # measured (paper Eq. 4)
    p: float = 1.0       # PDE residual (Eq. 5)
    bc: float = 1.0      # far-field Dirichlet (Eq. 6)
    ic: float = 0.0      # NOT in paper: initial condition
    wall: float = 0.0    # NOT in paper: zero-flux wellbore


@dataclass(frozen=True)
class SamplingConfig:
    n_p: int = 5000
    n_bc: int = 200
    n_ic: int = 200
    n_wall: int = 200
    method: str = "uniform"      # "uniform" | "sobol"
    r_spacing: str = "linear"    # "linear" | "log"
    resample: bool = False       # new collocation points every get()
    seed: int = 0


class Sampler:
    """Builds the training batch for one well. All times are returned in the
    model's time unit ('s', 'h', or 'h_raw'); radii in metres; T in deg C."""

    def __init__(self, case: WellCase, well: WellData, time_unit: str = "s",
                 cfg: SamplingConfig | None = None, ell_mult: float = 1.0):
        assert time_unit in ("s", "h", "h_raw")
        self.case, self.well = case, well
        self.cfg = SamplingConfig() if cfg is None else cfg
        self.t_mult = H_TO_S if time_unit == "s" else 1.0
        self.ell_mult = ell_mult
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        self._cache = None
        self._data = self._make_data()

    # --- helpers ---
    @staticmethod
    def _full(v, n):
        return torch.full((n, 1), float(v), dtype=DT)

    def _u(self, n, dim):
        if self.cfg.method == "uniform":
            return torch.rand(n, dim, generator=self.gen, dtype=DT)
        if self.cfg.method == "sobol":
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen))
            return torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(n, dtype=DT)
        raise ValueError(self.cfg.method)

    def _r(self, u):
        d = self.case.domain
        if self.cfg.r_spacing == "linear":
            return d.r_w + u * (d.r_max - d.r_w)
        if self.cfg.r_spacing == "log":
            return d.r_w * torch.exp(u * math.log(d.r_max / d.r_w))
        raise ValueError(self.cfg.r_spacing)

    def _t(self, u):
        return u * self.case.t_max_h * self.t_mult

    def _make_data(self):
        w = self.well
        t = torch.tensor(w.t_h, dtype=DT).reshape(-1, 1) * self.t_mult
        T = torch.tensor(w.bht, dtype=DT).reshape(-1, 1)
        return dict(r=self._full(self.case.domain.r_w, len(w.t_h)), t=t, T=T)

    def _fresh(self):
        c, case, d = self.cfg, self.case, self.case.domain
        u = self._u(c.n_p, 2)
        p = dict(r=self._r(u[:, 0:1]), t=self._t(u[:, 1:2]))
        bc = dict(r=self._full(d.r_max, c.n_bc), t=self._t(self._u(c.n_bc, 1)),
                  T=self._full(case.sft, c.n_bc))
        r_ic = self._r(self._u(c.n_ic, 1))
        T_ic = torch.tensor(gaussian_ic(r_ic.numpy().ravel(), case, ell_mult=self.ell_mult),
                            dtype=DT).reshape(-1, 1)
        ic = dict(r=r_ic, t=self._full(0.0, c.n_ic), T=T_ic)
        wall = dict(r=self._full(d.r_w, c.n_wall), t=self._t(self._u(c.n_wall, 1)))
        return dict(m=self._data, p=p, bc=bc, ic=ic, wall=wall)

    def get(self):
        if self.cfg.resample or self._cache is None:
            self._cache = self._fresh()
        return self._cache


def wall_flux(model, r, t):
    """T_r at the given points (autograd)."""
    r = r.detach().requires_grad_(True)
    T = model(r, t.detach())
    return torch.autograd.grad(T, r, grad_outputs=torch.ones_like(T), create_graph=True)[0]


def compute_losses(model, batch, w: LossWeights = LossWeights()):
    """model needs __call__(r, t) and .residual(r, t). Terms with weight 0 are skipped."""
    z = torch.zeros((), dtype=DT)
    parts = dict(m=z, p=z, bc=z, ic=z, wall=z)
    if w.m:
        b = batch["m"]; parts["m"] = ((model(b["r"], b["t"]) - b["T"]) ** 2).mean()
    if w.p:
        b = batch["p"]; parts["p"] = (model.residual(b["r"], b["t"]) ** 2).mean()
    if w.bc:
        b = batch["bc"]; parts["bc"] = ((model(b["r"], b["t"]) - b["T"]) ** 2).mean()
    if w.ic:
        b = batch["ic"]; parts["ic"] = ((model(b["r"], b["t"]) - b["T"]) ** 2).mean()
    if w.wall:
        b = batch["wall"]; parts["wall"] = (wall_flux(model, b["r"], b["t"]) ** 2).mean()
    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, parts

