import numpy as np
import System_Parameters as Para
import MPC


def calculate_depth(v_x, v_y, force):
    depth = np.zeros(len(v_x))
    for i in range(len(depth)):
        depth[i] = Para.calculate_depth(v_x[i], v_y[i], force[i])
    return depth


def run_openloop_mpc(x0, xf, horizon, ts):
    Para.set_horizon(horizon)
    Para.set_time_step(ts)
    Para.update_system_para()
    Para.set_initial_state(x0)
    Para.set_final_state(xf)
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    return feasibility, x_opt, u_opt


def find_smallest_horizon(x0: object, xf: object, ts: object) -> object:
    horizon = 1
    while True:
        horizon = horizon * 2
        Para.set_horizon(horizon)
        Para.set_time_step(ts)
        Para.set_initial_state(x0)
        Para.set_final_state(xf)
        Para.update_system_para()
        [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
        print('Horizon ' + str(horizon) + ', feasibility: ', feasibility)
        if feasibility:
            break

    lower_lim = horizon / 2 + 1
    upper_lim = horizon
    horizon = int((lower_lim + upper_lim) / 2)
    while True:
        if lower_lim == upper_lim:
            break
        Para.set_horizon(horizon)
        Para.set_time_step(ts)
        Para.set_initial_state(x0)
        Para.set_final_state(xf)
        Para.update_system_para()
        [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
        print('Horizon ' + str(horizon) + ', feasibility: ', feasibility)
        if feasibility:
            upper_lim = horizon
            horizon = int((lower_lim + upper_lim) / 2)
        else:
            lower_lim = horizon + 1
            horizon = int((lower_lim + upper_lim) / 2)
    print('Smallest Horizon for MPC: ', lower_lim)
    return lower_lim
