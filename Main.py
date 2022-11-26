import System_Parameters as Para
import MPC
import numpy as np
import PD_controller as Pd


if __name__ == "__main__":
    x0 = np.array([0, 0, 0, 0, 0])
    Para.set_horizon(20)
    Para.set_time_step(20)
    Para.set_initial_state(x0)
    Para.update_system_para()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
