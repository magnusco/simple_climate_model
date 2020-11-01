import numpy as np
import scipy.optimize
from dataclasses import dataclass
from typing import Callable

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


def calculate_equilibrium_temp(p: TableParameters, A: np.ndarray, E: np.ndarray):
    def F(temperature):
        X = np.concatenate((np.power(temperature, 4), temperature))
        return np.matmul(A, X) - E

    return scipy.optimize.fsolve(F, np.array([300, 300]), xtol=1e-9)


def print_energy_flows(
    p: TableParameters,
    gamma_0,
    gamma_1,
    lambda_0,
    lambda_1,
    mu_0,
    mu_1,
    equlibrium_temperature,
):
    P_E_in = gamma_0 * p.P_s
    P_A_in = gamma_1 * p.P_s
    R_E_in = (lambda_0 * p.sigma * p.e_E * equlibrium_temperature[0] ** 4 
        + mu_0 * p.sigma * p.e_A * equlibrium_temperature[1] ** 4)
    
    R_A_in = (lambda_1 * p.sigma * p.e_E * equlibrium_temperature[0] ** 4 
        + mu_1 * p.sigma * p.e_A * equlibrium_temperature[1] ** 4)
    
    R_E_0 = p.sigma * p.e_E * equlibrium_temperature[0] ** 4
    R_A_0 = p.sigma * p.e_A * equlibrium_temperature[1] ** 4
    P_lat = (p.alpha + p.beta) * (equlibrium_temperature[0] - equlibrium_temperature[1])

    print("P_E_in, P_A_in: ", P_E_in, P_A_in)
    print("R_E_in, R_A_in: ", R_E_in, R_A_in)
    print("R_E_0, R_A_0: ", R_E_0, R_A_0)
    print("P_lat: ", P_lat)


def climate_model_1(p: TableParameters, flow_info: bool):
    # Clouds are below atmosphere, energy absorbed by clouds are transmitted to atmosphere.

    gamma_0 = ((1 - p.r_SM) * (1 - p.C_C * p.r_SC) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.a_SC))
    lambda_0 = p.C_C * p.r_LC * (1 - p.r_LE)
    mu_0 = p.f_A * (1 - p.C_C * p.r_LC) * (1 - p.C_C * p.a_LC) * (1 - p.r_LE)

    gamma_1 = ((1 - p.r_SM) * p.a_SW + (1 - p.r_SM) * (1 - p.a_SW) * p.a_O3 
    + (1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * p.C_C * p.a_SC)

    lambda_1 = (1 - p.C_C * p.r_LC) * p.C_C * p.a_LC + (1 - p.C_C * p.r_LC) * (1 - p.C_C * p.a_LC) * p.a_LW
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

    equlibrium_temperature = calculate_equilibrium_temp(p, A, E)

    if flow_info == True:
        print_energy_flows(
            p, gamma_0, gamma_1, lambda_0, lambda_1, mu_0, mu_1, equlibrium_temperature
        )

    return equlibrium_temperature


def climate_model_2(p: TableParameters, flow_info: bool):
    # New reflection and absorption coefficients to account for clouds in one-dim layer.

    a_LAC = (0.5 * p.a_LW + (1 - 0.5 * p.a_LW) * (1 - p.r_LC) * p.a_LC 
        + (1 - 0.5 * p.a_LW) * (1 - p.r_LC) * (1 - p.a_LC) * 0.5 * p.a_LW )
    r_LAC = (1 - 0.5 * p.a_LW) * p.r_LC

    gamma_0 = ((1 - p.r_SM) * (1 - p.C_C * p.r_SC) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.a_SC))
    lambda_0 = p.C_C * r_LAC * (1 - p.r_LE)
    mu_0 = p.f_A * (1 - p.r_LE)

    gamma_1 = ((1 - p.r_SM) * p.a_SW + (1 - p.r_SM) * (1 - p.a_SW) * p.a_O3
    + (1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * p.C_C * p.a_SC)
    lambda_1 = p.C_C * (1 - r_LAC) * a_LAC + (1 - p.C_C) * p.a_LW
    mu_1 = 0


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

    equlibrium_temperature = calculate_equilibrium_temp(p, A, E)

    if flow_info == True:
        print_energy_flows(
            p, gamma_0, gamma_1, lambda_0, lambda_1, mu_0, mu_1, equlibrium_temperature
        )

    return equlibrium_temperature


def find_derivatives(climate_model: Callable, h: float, change):
    base_params = TableParameters()
    p_a_SW = TableParameters(a_SW=0.1451 + h)
    p_a_LW = TableParameters(a_LW=0.8358 + h)
    p_r_SM = TableParameters(r_SM=0.1065 + h)
    p_r_SE = TableParameters(r_SE=0.1700 + h)
    p_a_O3 = TableParameters(a_O3=0.0800 + h)
    p_r_LC = TableParameters(r_LC=0.1950 + h)
    
    a_SW_deriv = (climate_model(p_a_SW, False) - climate_model(base_params, False)) / h
    a_LW_deriv = (climate_model(p_a_LW, False) - climate_model(base_params, False)) / h
    r_SM_deriv = (climate_model(p_r_SM, False) - climate_model(base_params, False)) / h
    r_SE_deriv = (climate_model(p_r_SE, False) - climate_model(base_params, False)) / h
    a_O3_deriv = (climate_model(p_a_O3, False) - climate_model(base_params, False)) / h
    r_LC_deriv = (climate_model(p_r_LC, False) - climate_model(base_params, False)) / h
    print(climate_model(p_a_LW, False))
    print(climate_model(base_params, False))

    print("a_SW derivative:  ", a_SW_deriv, "  Change:  ", a_SW_deriv * change)
    print("a_LW derivative:  ", a_LW_deriv, "  Change:  ", a_LW_deriv * change)
    print("r_SM derivative:  ", r_SM_deriv, "  Change:  ", r_SM_deriv * change)
    print("r_SE derivative:  ", r_SE_deriv, "  Change:  ", r_SE_deriv * change)
    print("a_O3 derivative:  ", a_O3_deriv, "  Change:  ", a_O3_deriv * change)
    print("r_LC derivative:  ", r_LC_deriv, "  Change:  ", r_LC_deriv * change)



if __name__ == "__main__":

    base_params = TableParameters()
    cels_norm = 273.15

    # print(climate_model_1(base_params, True) - cels_norm)
    find_derivatives(climate_model_2, 1e-3, 0.02)
