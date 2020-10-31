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
    e_A: float = 2 * 0.875
    f_A: float = 0.618
    alpha: float = 3
    beta: float = 4
    sigma: float = 5.670e-8


def climate_model_1(p: TableParameters, flow_info: bool):

    gamma_0 = (
        (1 - p.r_SM)
        * (1 - p.C_C * p.r_SC)
        * (1 - p.a_SW)
        * (1 - p.a_O3)
        * (1 - p.C_C * p.a_SC)
    )
    lambda_0 = p.C_C * p.r_LC * (1 - p.r_LE)
    mu_0 = p.f_A * (1 - p.C_C * p.r_LC) * (1 - p.C_C * p.a_LC) * (1 - p.r_LE)

    gamma_1 = (
        (1 - p.r_SM) * p.a_SW
        + (1 - p.r_SM) * (1 - p.a_SW) * p.a_O3
        + (1 - p.r_SM)
        * (1 - p.a_SW)
        * (1 - p.a_O3)
        * (1 - p.C_C * p.r_SC)
        * p.C_C
        * p.a_SC
    )
    lambda_1 = (1 - p.C_C * p.r_LC) * p.C_C * p.a_LC + (1 - p.C_C * p.r_LC) * (
        1 - p.C_C * p.a_LC
    ) * p.a_LW
    mu_1 = p.f_A * p.C_C * p.r_LC * p.a_LW + p.f_A * p.C_C * p.r_LC * p.a_LW

    a_0 = p.sigma * p.e_E * (1 - lambda_0)
    b_0 = -p.sigma * p.e_A * mu_0
    c_0 = p.alpha + p.beta
    d_0 = -(p.alpha + p.beta)
    e_0 = gamma_0 * p.P_s

    a_1 = -p.sigma * p.e_E * lambda_1
    b_1 = p.sigma * p.e_A * (1 - mu_1)
    c_1 = -(p.alpha + p.beta)
    d_1 = p.alpha + p.beta
    e_1 = gamma_1 * p.P_s

    A = np.array([[a_0, b_0, c_0, d_0], [a_1, b_1, c_1, d_1]])
    E = np.array([e_0, e_1])

    def F(temperature):
        X = np.concatenate((np.power(temperature, 4), temperature))
        return np.matmul(A, X) - E

    equlibrium_temperature = scipy.optimize.fsolve(F, np.array([300, 300]), xtol=1e-9)

    if flow_info == True:
        P_E_in = gamma_0 * p.P_s
        P_A_in = gamma_1 * p.P_s
        R_E_in = (
            lambda_0 * p.sigma * p.e_E * equlibrium_temperature[0] ** 4
            + mu_0 * p.sigma * p.e_A * equlibrium_temperature[1] ** 4
        )
        R_A_in = (
            lambda_1 * p.sigma * p.e_E * equlibrium_temperature[0] ** 4
            + mu_1 * p.sigma * p.e_A * equlibrium_temperature[1] ** 4
        )
        R_E_0 = p.sigma * p.e_E * equlibrium_temperature[0] ** 4
        R_A_0 = p.sigma * p.e_A * equlibrium_temperature[1] ** 4
        P_lat = (p.alpha + p.beta) * (
            equlibrium_temperature[0] - equlibrium_temperature[1]
        )
        print("P_E_in, P_A_in: ", P_E_in, P_A_in)
        print("R_E_in, R_A_in: ", R_E_in, R_A_in)
        print("R_E_0, R_A_0: ", R_E_0, R_A_0)
        print("P_lat: ", P_lat)

    return equlibrium_temperature


if __name__ == "__main__":

    p = TableParameters()
    cels_norm = 273.15

    print(climate_model_1(p, True) - cels_norm)

