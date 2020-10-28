import numpy as np
import scipy.optimize
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
    
    a_0 = p.e_E * p.sigma - p.C_C * p.r_LE * (1 - p.r_LE) * p.e_E * p.sigma
    b_0 = p.f_A * (p.r_LE - 1) * p.e_A * p.sigma
    c_0 = p.alpha
    d_0 = - p.beta
    e_0 = (1 - p.r_SE - p.r_SM - p.C_C * p.r_SC) * (1 - p.a_SW - p.a_O3 - p.C_C * p.a_SC) * p.P_s

    a_1 = p.e_E * p.sigma * (p.C_C * p.r_LC - 1)
    b_1 = p.e_A * p.sigma + (1 - p.f_A) * p.e_A * p.sigma
    c_1 = - p.alpha
    d_1 = p.beta
    e_1 = (1 - p.r_SE - p.r_SM - p.C_C * p.r_SC) * (p.a_SW + p.a_O3 + p.C_C * p.a_SC) * p.P_s

    A, E = np.array([[a_0, b_0, c_0, d_0], [a_1, b_1, c_1, d_1]]), np.array([e_0, e_1])


    def F(temperature: np.ndarray):
        X = np.concatenate((np.power(temperature, 4), temperature))
        return np.matmul(A, X) - E


    
    # print((1 - p.C_C * p.r_SC - p.r_SE - p.r_SM) * (1 - p.a_O3 - p.a_SW - p.C_C * p.a_SC))

    equilib_temp = scipy.optimize.anderson(F, np.array([280, 280]), f_tol = 1e-9)
    print(equilib_temp - 273.15)
    

    
