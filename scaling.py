import numpy as np
from config import WellCase, H_TO_S

class Scaler:
    """Maps physical (r [m], t [h], T [C]) <-> dimensionless (r*, t*, theta)."""

    def __init__(self, case: WellCase, D: float | None = None):
        self.case = case
        self.D = case.rock.D if D is None else D        # m^2/s
        self.r_w = case.domain.r_w
        self.dT = case.sft - case.t0_bht
        self.T0 = case.t0_bht

    # lengths
    def r_to_star(self, r):   return np.asarray(r) / self.r_w
    def r_from_star(self, rs): return np.asarray(rs) * self.r_w
    # time (hours in, dimensionless out)
    def t_to_star(self, t_h): return self.D * np.asarray(t_h) * H_TO_S / self.r_w**2
    def t_from_star(self, ts): return np.asarray(ts) * self.r_w**2 / (self.D * H_TO_S)
    # temperature
    def T_to_theta(self, T):  return (np.asarray(T) - self.T0) / self.dT
    def theta_to_T(self, th): return np.asarray(th) * self.dT + self.T0

    @property
    def r_star_max(self): return self.case.domain.r_max / self.r_w
    @property
    def t_star_max(self): return float(self.t_to_star(self.case.t_max_h))