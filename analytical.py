import numpy as np
from data import WellData

def time_function(method: str, t_h, t_c_h):
    t = np.asarray(t_h, float)
    if method == "horner":
        return np.log((t + t_c_h) / t)
    if method == "ascencio":
        return 1.0 / np.sqrt(t)
    raise ValueError(method)

def estimate_sft(w: WellData, method: str, last_n: int | None = None) -> dict:
    """OLS of BHT on the method's time function; SFT = intercept (x -> 0, i.e. dt -> inf)."""
    m = w.regression_mask()
    t, T = w.t_h[m], w.bht[m]
    if last_n is not None:
        t, T = t[-last_n:], T[-last_n:]
    x = time_function(method, t, w.t_c_h)
    slope, icpt = np.polyfit(x, T, 1)
    That = icpt + slope * x
    ss_res = np.sum((T - That) ** 2)
    ss_tot = np.sum((T - T.mean()) ** 2)
    r = float(np.corrcoef(x, T)[0, 1])          # negative for both methods
    return dict(sft=float(icpt), slope=float(slope), r2=float(1 - ss_res / ss_tot),
                r=abs(r), n=len(t))

