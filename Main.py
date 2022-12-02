import System_Parameters as Para
import MPC
import numpy as np
import Main_func as Func
import PD_controller as Pd


if __name__ == "__main__":
    x0 = np.array([0, 0, 0, 0, 0])
    xf = np.array([20, 20, 0, 0, 4])
    Para.set_horizon(5)
    Para.set_time_step(20)
    Para.set_initial_state(x0)
    Para.set_final_state(xf)
    Para.update_system_para()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    print(feasibility)
    print('xOpt =', x_opt)
    print('uOpt = ', u_opt)
    _input = Func.input_modify(u_opt)
