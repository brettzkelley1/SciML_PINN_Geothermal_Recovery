import numpy as np
from scipy.linalg import solve_banded
from config import WellCase, H_TO_S
from ic import gaussian_ic

def solve_fd(case: WellCase, n_r: int = 400, n_t: int = 2000, D: float | None = None,
             ic_fn=None, wall_bc: str = "neumann", bht_fn=None,
             theta: float = 0.5, n_out: int = 200):
    """
    dT/dt = D (T_rr + T_r / r) on r in [r_w, r_max], solved on s = ln(r/r_w):
        dT/dt = D / (r_w^2 e^{2s}) * T_ss
    Far BC: Dirichlet T = case.sft.
    Wall BC: 'neumann' (zero flux) or 'dirichlet' (T = bht_fn(t_hours)).
    Returns dict with r [m], t_h [h], T [n_out+1, n_r] (deg C), bht = T[:, 0].
    """
    D = case.rock.D if D is None else D
    r_w, r_max = case.domain.r_w, case.domain.r_max
    s = np.linspace(0.0, np.log(r_max / r_w), n_r)
    h = s[1] - s[0]
    r = r_w * np.exp(s)
    c = D / (r**2) / h**2                       # c_i = D / (r_i^2 h^2); r_w^2 e^{2s} = r^2

    # operator L: (L T)_i = c_i (T_{i+1} - 2 T_i + T_{i-1})
    lo = c.copy(); di = -2.0 * c; up = c.copy()
    up[0] = 2.0 * c[0]; lo[0] = 0.0             # Neumann at wall via ghost node T_{-1} = T_1
    lo[-1] = 0.0; di[-1] = 0.0; up[-1] = 0.0    # far boundary row handled as Dirichlet

    def apply_L(T):
        out = di * T
        out[:-1] += up[:-1] * T[1:]
        out[1:] += lo[1:] * T[:-1]
        return out

    T = (gaussian_ic(r, case, D=D) if ic_fn is None else ic_fn(r)).astype(float)
    T[-1] = case.sft
    if wall_bc == "dirichlet":
        T[0] = bht_fn(0.0)

    def step(T, dt, t_new_s):
        ab = np.zeros((3, n_r))
        a_lo, a_di, a_up = -theta * dt * lo, 1.0 - theta * dt * di, -theta * dt * up
        a_lo[-1] = 0.0; a_di[-1] = 1.0                       # far Dirichlet row
        if wall_bc == "dirichlet":
            a_di[0] = 1.0; a_up[0] = 0.0                     # wall Dirichlet row
        ab[0, 1:] = a_up[:-1]; ab[1, :] = a_di; ab[2, :-1] = a_lo[1:]
        rhs = T + (1.0 - theta) * dt * apply_L(T)
        rhs[-1] = case.sft
        if wall_bc == "dirichlet":
            rhs[0] = bht_fn(t_new_s / H_TO_S)
        return solve_banded((1, 1), ab, rhs)

    t_end = case.t_max_h * H_TO_S
    dt = t_end / n_t
    out_every = max(1, n_t // n_out)
    t_out, T_out = [0.0], [T.copy()]
    t = 0.0
    for k in range(n_t):
        if k == 0:                                           # Rannacher startup: 4 BE substeps
            th_save, theta = theta, 1.0
            for _ in range(4):
                t += dt / 4; T = step(T, dt / 4, t)
            theta = th_save
        else:
            t += dt; T = step(T, dt, t)
        if (k + 1) % out_every == 0:
            t_out.append(t / H_TO_S); T_out.append(T.copy())
    return dict(r=r, t_h=np.array(t_out), T=np.array(T_out), bht=np.array(T_out)[:, 0])