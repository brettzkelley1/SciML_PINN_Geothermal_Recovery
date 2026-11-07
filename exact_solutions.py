import numpy as np
from scipy.special import j0, j1, y0, y1
from scipy.optimize import brentq


def _phi(inner: str):
    if inner == "neumann":
        return lambda lam, r, r_w: j0(lam * r) * y1(lam * r_w) - y0(lam * r) * j1(lam * r_w)
    if inner == "dirichlet":
        return lambda lam, r, r_w: j0(lam * r) * y0(lam * r_w) - y0(lam * r) * j0(lam * r_w)
    raise ValueError(inner)


def eigenvalues(n_modes: int, r_w: float, r_max: float, inner: str = "neumann"):
    """First n_modes positive roots of phi(lam, r_max) = 0."""
    phi = _phi(inner)
    lam_hi = (n_modes + 2) * np.pi / (r_max - r_w) * 1.5
    grid = np.arange(1e-3, lam_hi, 1e-3)          # root spacing ~ pi/(r_max-r_w) ~ 0.5
    f = phi(grid, r_max, r_w)
    idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = [brentq(lambda l: phi(l, r_max, r_w), grid[i], grid[i + 1], xtol=1e-14) for i in idx]
    assert len(roots) >= n_modes, "not enough roots found"
    return np.array(roots[:n_modes])


class ExactSolution:
    """T(r, t_s) = sft + sum_k scale_k * phi_k(r) * exp(-D lam_k^2 t_s).  t in SECONDS."""

    def __init__(self, case, modes, amps, inner="neumann", D=None):
        self.case, self.inner = case, inner
        self.D = case.rock.D if D is None else D
        self.r_w, self.r_max = case.domain.r_w, case.domain.r_max
        self.modes = list(modes)
        self.lams = eigenvalues(max(self.modes), self.r_w, self.r_max, inner)[[m - 1 for m in self.modes]]
        self._phi = _phi(inner)
        rr = np.linspace(self.r_w, self.r_max, 5000)
        # scale each mode so that max|A_k phi_k| = amps[k] (deg C)
        self.scales = np.array([a / np.max(np.abs(self._phi(l, rr, self.r_w)))
                                for a, l in zip(amps, self.lams)])

    def tau_s(self, k: int = 0) -> float:
        """e-folding time of the k-th listed mode, seconds."""
        return 1.0 / (self.D * self.lams[k] ** 2)

    def __call__(self, r, t_s):
        r = np.asarray(r, float); t_s = np.asarray(t_s, float)
        out = np.zeros(np.broadcast(r, t_s).shape)
        for lam, sc in zip(self.lams, self.scales):
            out = out + sc * self._phi(lam, r, self.r_w) * np.exp(-self.D * lam**2 * t_s)
        return self.case.sft + out

    