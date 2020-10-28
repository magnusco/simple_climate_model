import numpy as np
from dataclasses import dataclass

@dataclass
class TableParameters:
    P_s: float = 341.3
    C_C: float = 0.66
    r_SM: float = 0.1065
    r_SC: float = 0.22
    r_SE: float = 0.17
    a_O3: float = 0.08
    a_SC: float = 0.1239
    a_SW: float = 0.1451
    r_LC: float = 0.195
    r_LE: float = 0.0
    a_LC: float = 0.622
    a_LW: float = 0.8358
    e_E: float = 1.0
    e_A: float = 0.875
    f_A: float = 0.618
    alpha: float = 3
    beta: float = 4
    sigma: float = 5.670e-8


if __name__ == "__main__":

    p = TableParameters()

    R_E_0 = 396
    lw_through_atmospheric_window_frac = (1 - p.C_C * p.a_LC) * (1 - p.a_LW)

    P_S_0 = 341
    
    P_E_in_frac = (1 - p.r_SM) * (1 - p.C_C * p.r_SC) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.a_SC)
    P_A_in_frac = ((1 - p.r_SM) * p.a_SW + (1 - p.r_SM) * (1 - p.a_SW) * p.a_O3 + (1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * p.C_C * p.a_SC)

    R_E_in_frac_atm = p.f_A * (1 - p.r_LE)
    R_E_in_frac_earth = p.C_C * p.r_LC * (1 - p.r_LE)
    R_A_in_frac = (1 - p.C_C * p.a_LC) * p.C_C * p.a_LC + (1 - p.C_C * p.a_LC) * (1 - p.C_C * p.a_LC) * p.a_LW

    print(P_E_in_frac, P_A_in_frac)
    print(R_E_in_frac_atm, R_E_in_frac_earth, R_A_in_frac)
    