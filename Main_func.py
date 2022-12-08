import numpy as np
import System_Parameters as Para
<<<<<<< HEAD


def input_modify(u_opt):
    ts = Para.get_time_step()
    _input = np.zeros((5, u_opt.shape[1]))
    _input[4, 0] = u_opt[2, 0]
    _input[2, 0] = u_opt[0, 0] * ts
    _input[3, 0] = u_opt[1, 0] * ts
    _input[0, 0] = 0.5 * u_opt[0, 0] * ts ** 2
    _input[1, 0] = 0.5 * u_opt[1, 0] * ts ** 2
    for i in range(1, u_opt.shape[0]):
        _input[4, i] = u_opt[2, i]
        _input[2, i] = u_opt[0, i] * ts + _input[2, i - 1]
        _input[3, i] = u_opt[1, i] * ts + _input[3, i - 1]
        _input[0, i] = 0.5 * u_opt[0, i] * ts ** 2 + _input[2, i - 1] * ts + _input[0, i - 1]
        _input[1, i] = 0.5 * u_opt[1, i] * ts ** 2 + _input[3, i - 1] * ts + _input[1, i - 1]
    return _input
=======
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
>>>>>>> Zedai-Yang
