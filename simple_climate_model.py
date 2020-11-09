import numpy as np
import scipy.optimize
from dataclasses import dataclass, replace
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

    gamma_0 = ((1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * (1 - p.C_C * p.a_SC)) * (1 - p.r_SE)
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
        + (1 - 0.5 * p.a_LW) * (1 - p.r_LC) * (1 - p.a_LC) * p.a_LW )
    r_LAC = (1 - 0.5 * p.a_LW)**2 * p.r_LC

    gamma_0 = ((1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * (1 - p.C_C * p.a_SC)) * (1 - p.r_SE)
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

def climate_model_3(p: TableParameters, flow_info: bool):
    # Custom coefficients to reproduce energy flows.

    a_LAC = (0.5 * p.a_LW + (1 - 0.5 * p.a_LW) * (1 - p.r_LC) * p.a_LC
        + (1 - 0.5 * p.a_LW) * (1 - p.r_LC) * (1 - p.a_LC) * p.a_LW )
    r_LAC = (1 - 0.5 * p.a_LW)**2 * p.r_LC

    gamma_0 = ((1 - p.r_SM) * (1 - p.a_SW) * (1 - p.a_O3) * (1 - p.C_C * p.r_SC) * (1 - p.C_C * p.a_SC)) * (1 - p.r_SE)
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


def find_derivatives(climate_model: Callable, base_params: TableParameters, h: float, change):

    p_r_SM = replace(base_params, r_SM = base_params.r_SM + h)
    p_r_SC = replace(base_params, r_SC=base_params.r_SC + h)
    p_r_SE = replace(base_params, r_SE=base_params.r_SE + h)
    p_a_O3 = replace(base_params, a_O3=base_params.a_O3 + h)
    p_a_SC = replace(base_params, a_SC=base_params.a_SC + h)
    p_a_SW = replace(base_params, a_SW=base_params.a_SW + h)
    p_r_LC = replace(base_params, r_LC=base_params.r_LC + h)
    p_r_LE = replace(base_params, r_LE=base_params.r_LE + h)
    p_a_LC = replace(base_params, a_LC=base_params.a_LC + h)
    p_a_LW = replace(base_params, a_LW=base_params.a_LW + h)
    p_f_A = replace(base_params, f_A=base_params.f_A + h)
    p_alpha = replace(base_params, alpha=base_params.alpha + h)
    p_beta = replace(base_params, beta=base_params.beta + h)    

    r_SM_deriv = (climate_model(p_r_SM, False) - climate_model(base_params, False)) / h
    r_SC_deriv = (climate_model(p_r_SC, False) - climate_model(base_params, False)) / h
    r_SE_deriv = (climate_model(p_r_SE, False) - climate_model(base_params, False)) / h
    a_O3_deriv = (climate_model(p_a_O3, False) - climate_model(base_params, False)) / h
    a_SC_deriv = (climate_model(p_a_SC, False) - climate_model(base_params, False)) / h
    a_SW_deriv = (climate_model(p_a_SW, False) - climate_model(base_params, False)) / h
    r_LC_deriv = (climate_model(p_r_LC, False) - climate_model(base_params, False)) / h
    r_LE_deriv = (climate_model(p_r_LE, False) - climate_model(base_params, False)) / h
    a_LC_deriv = (climate_model(p_a_LC, False) - climate_model(base_params, False)) / h
    a_LW_deriv = (climate_model(p_a_LW, False) - climate_model(base_params, False)) / h
    f_A_deriv = (climate_model(p_f_A, False) - climate_model(base_params, False)) / h
    alpha_deriv = (climate_model(p_alpha, False) - climate_model(base_params, False)) / h
    beta_deriv = (climate_model(p_beta, False) - climate_model(base_params, False)) / h

    print("r_SM derivative:  ", r_SM_deriv, "  Change:  ", r_SM_deriv * change)
    print("r_SC derivative:  ", r_SC_deriv, "  Change:  ", r_SC_deriv * change)
    print("r_SE derivative:  ", r_SE_deriv, "  Change:  ", r_SE_deriv * change)
    print("a_O3 derivative:  ", a_O3_deriv, "  Change:  ", a_O3_deriv * change)
    print("a_SC derivative:  ", a_SC_deriv, "  Change:  ", a_SC_deriv * change)
    print("a_SW derivative:  ", a_SW_deriv, "  Change:  ", a_SW_deriv * change)
    print("r_LC derivative:  ", r_LC_deriv, "  Change:  ", r_LC_deriv * change)
    print("r_LE derivative:  ", r_LE_deriv, "  Change:  ", r_LE_deriv * change)
    print("a_LC derivative:  ", a_LC_deriv, "  Change:  ", a_LC_deriv * change)
    print("a_LW derivative:  ", a_LW_deriv, "  Change:  ", a_LW_deriv * change)
    print("f_A derivative:  ", f_A_deriv, "  Change:  ", f_A_deriv * change)
    print("alpha derivative:  ", alpha_deriv, "  Change:  ", alpha_deriv * change)
    print("beta derivative:  ", beta_deriv, "  Change:  ", beta_deriv * change)


if __name__ == "__main__":

    base_params = TableParameters()
    cels_norm = 273.15


    # Model 3 with custom coefficients
    custom_params = TableParameters(
        r_SC=base_params.r_SC - 0.02,
        a_SW=base_params.a_SW - 0.015,
        f_A=base_params.f_A + 0.012,
        a_LW=base_params.a_LW + 0.05,
        )


    find_derivatives(climate_model_2, custom_params, 1e-4, change=0.02)    
    print(climate_model_2(base_params, True) - cels_norm)