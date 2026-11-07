import numpy as np
from config import WellCase, H_TO_S

def gaussian_ic(r, case: WellCase, ell_mult: float = 1.0, D: float | None = None):
    """Cooled zone at shut-in: T(r_w) = t0_bht, relaxing to SFT over ell = ell_mult*sqrt(4 D t_c)."""
    D = case.rock.D if D is None else D
    ell = ell_mult * np.sqrt(4.0 * D * case.t_c_h * H_TO_S)
    r = np.asarray(r, float)
    return case.sft - (case.sft - case.t0_bht) * np.exp(-((r - case.domain.r_w) / ell) ** 2)

