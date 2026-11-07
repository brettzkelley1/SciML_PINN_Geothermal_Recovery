"""Well data transcribed from Bassam et al. (2010) Table 1 / Table 3
and Canilandi et al. (2025) Table 1 / Fig. 6.
Units: time [h] (shut-in time, Delta t), temperature [deg C], length [m]."""
from dataclasses import dataclass, field
import numpy as np
from config import WellCase

# name: (t_c [h], group, t [h] list, BHT [C] list)
_RAW = {
 "MXCO1": (2.5, "train", [6,12,18,24,30,36],
           [186.3,206.9,219.0,231.9,239.8,247.7]),
 "MXCO2": (2.5, "train", [6,12,18,24,30,36,42],
           [178.6,198.9,211.4,225.8,235.1,240.4,247.1]),
 "SGIL":  (3.0, "train", list(range(13)),
           [62.78,81.94,86.56,88.84,90.76,92.29,93.44,94.21,94.59,94.98,95.36,95.74,96.13]),
 "USAM":  (5.0, "train", [1,2,5,10,15,20,25,30,40,50,70,100,150,200],
           [138.75,139.55,140.89,142.03,142.72,143.21,143.60,143.92,144.42,144.81,145.40,146.04,146.76,147.27]),
 "PHIL":  (15.0, "train", [2.58+i for i in range(14)],
           [93,88,99,108,112,117,120,126,133,133,134,137,141,146]),
 "JAPN":  (5.0, "train", [5.5,6.5,7.5,8.5,9.5,12.5,15.5,18.5,24.5,72.5],
           [92.0,98.5,103.0,107.4,110.4,119.6,126.6,132.8,142.4,170.9]),
 "CH-A1": (5.0, "train", [6,12,26.57,47.1,95.27],
           [90,102,116,130,145]),
 "CH-A2": (5.0, "train", [6.0,12.0,26.6,47.1,95.3,190.5],
           [231,231,235,237,239,240]),
 "ITAL":  (5.0, "train",
           [5.7,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,
            15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0,21.5,22.0,22.5,23.0,24.0,25.0,26.0,27.0],
           [84.5,93.1,95.0,96.2,97.5,98.4,99.3,100.3,101.1,102.0,102.7,103.4,104.2,104.7,105.4,106.1,106.7,107.6,
            108.3,108.9,109.6,110.2,110.8,111.3,111.9,112.4,112.9,113.3,113.8,114.2,114.7,115.1,115.4,115.7,116.1,
            116.5,117.1,117.7,117.9,118.7]),
 "COST":  (4.0, "train", [8.5,10.0,16.5,34.5,38.0,75.5],
           [45.00,49.44,55.56,56.11,57.22,56.11]),
 # validation wells (true formation temperature known)
 "SHBE":  (5.0, "val", [2.5,5,7.5,10,15,20,30,40],
           [56.6,61.3,64.3,66.6,69.6,71.7,74.1,75.5]),
 "CLAH":  (5.0, "val", [2,4,6,8,10,12,14,16,18,20,22,24,30,40,50],
           [91.7,98.1,102.4,105.5,107.9,109.8,111.3,112.5,113.6,114.4,115.2,115.8,117.1,118.4,119.1]),
 "CH-A4": (5.0, "val", [6,12,26.57,47.1,95.27],
           [85,94,122,139,152]),
 "KJ-21": (2.5, "val", [2,30,52,96,242,556],
           [50,125,158,184,218,225]),
}
_M_EXPECTED = {"MXCO1":6,"MXCO2":7,"SGIL":13,"USAM":14,"PHIL":14,"JAPN":10,"CH-A1":5,"CH-A2":6,
               "ITAL":40,"COST":6,"SHBE":8,"CLAH":15,"CH-A4":5,"KJ-21":6}

# extra properties reported in Bassam Table 1 footnote
_EXTRA = {
 "SGIL":  dict(r_w=0.10028, kappa=2.25419, cp=878.64, rho=2306.6),
 "COST":  dict(kappa=4.97),
 "SHBE":  dict(r_w=0.108, tft=80.0),
 "CLAH":  dict(r_w=0.108, tft=120.0),
 "CH-A4": dict(tft=169.0),
 "KJ-21": dict(tft=240.0),
}

# Canilandi Table 1: (SFT Ascencio, SFT Horner, R2 Ascencio, R2 Horner)
PAPER_TABLE1 = {
 "MXCO1": (283.30,253.42,0.98,0.96), "MXCO2": (282.71,251.30,0.97,0.95),
 "SGIL":  (101.73, 98.45,0.99,0.99), "USAM":  (146.04,145.19,0.91,0.89),
 "PHIL":  (178.81,171.43,0.95,0.97), "JAPN":  (193.33,167.40,0.99,0.97),
 "CH-A1": (155.91,139.79,0.97,0.94), "CH-A2": (241.38,238.95,0.95,0.91),
 "COST":  ( 64.01, 59.85,0.89,0.92), "ITAL":  (142.39,127.86,0.99,0.99),
}

# Bassam Table 3: (analytical mean SFT, sd, ANN SFT, sd)
BASSAM_TABLE3 = {
 "MXCO1": (263,15,256,20), "MXCO2": (262,18,253,15), "USAM": (146,1,139,5),
 "ITAL":  (132,10,133,8),  "PHIL":  (197,28,185,25), "JAPN": (176,18,180,11),
 "CH-A1": (147,5,151,7),   "CH-A2": (241,18,243,15), "COST": (57,3,56,2),
 "SGIL":  (100,3,102,9),
}

# Approx. simulation windows [h] read from the last timestep labels in Canilandi Fig. 6
PAPER_TMAX_H = {
 "MXCO1":350,"MXCO2":350,"SGIL":100,"USAM":250,"PHIL":130,"JAPN":220,"CH-A1":250,
 "CH-A2":250,"ITAL":270,"COST":755,"SHBE":200,"CLAH":200,"CH-A4":350,"KJ-21":650,
}

@dataclass(frozen=True)
class WellData:
    name: str
    group: str                 # "train" or "val"
    t_c_h: float
    t_h: np.ndarray            # shut-in times, h
    bht: np.ndarray            # deg C
    extra: dict = field(default_factory=dict)

    @property
    def tft(self): return self.extra.get("tft")

    def regression_mask(self):
        """Horner (ln) and Ascencio (1/sqrt) are singular at dt = 0 (SGIL has one)."""
        return self.t_h > 0

def load_well(name: str, drop_dt0: bool = False) -> WellData:
    t_c, grp, t, T = _RAW[name]
    w = WellData(name, grp, t_c, np.array(t, float), np.array(T, float), dict(_EXTRA.get(name, {})))
    assert len(w.t_h) == len(w.bht) == _M_EXPECTED[name], f"{name}: length mismatch"
    assert np.all(np.diff(w.t_h) > 0), f"{name}: times not increasing"
    if drop_dt0 and (w.t_h <= 0).any():
        k = w.t_h > 0
        w = WellData(w.name, w.group, w.t_c_h, w.t_h[k], w.bht[k], w.extra)
    return w

def all_wells(group: str | None = None) -> dict[str, WellData]:
    return {n: load_well(n) for n, v in _RAW.items() if group in (None, v[1])}

def make_case(name: str, sft: float | None = None, method: str = "ascencio") -> WellCase:
    """Build a WellCase. sft defaults to Canilandi Table 1 (training wells only);
    for validation wells pass the SFT from your own step-3 estimator.
    t0_bht is a PROXY (first measured BHT) since true shut-in temperature is not
    reported (except SGIL). It is unused in paper mode."""
    w = load_well(name)
    if sft is None:
        if name not in PAPER_TABLE1:
            raise ValueError(f"{name}: no paper SFT; run step-3 estimators and pass sft=")
        sft = PAPER_TABLE1[name][0 if method == "ascencio" else 1]
    return WellCase(name=name, t_c_h=w.t_c_h, sft=float(sft),
                    t0_bht=float(w.bht[0]), t_max_h=float(PAPER_TMAX_H[name]))