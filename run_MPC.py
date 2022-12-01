import System_Parameters as Para
import MPC
import numpy as np
import PD_controller as Pd

def set_MPC_param(horizon, time_step):
    Para.set_horizon(horizon)
    Para.set_time_step(time_step)
    Para.update_system_para()


def run_MPC(x0):
    Para.set_initial_state(x0)
    Para.update_system_para()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    return u_opt


if __name__ == "__main__":
    set_MPC_param(horizon=20, time_step=20)
    u_opt = run_MPC(x0=np.array([0, 0, 0, 0, 0]))
    
    
