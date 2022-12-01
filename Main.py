import System_Parameters as Para
import MPC
import numpy as np
import PD_controller as Pd

def run_MPC(x0):
    Para.set_horizon(20)
    Para.set_time_step(20)
    Para.set_initial_state(x0)
    Para.update_system_para()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    return u_opt


if __name__ == "__main__":
    u_opt = run_MPC(np.array([0, 0, 0, 0, 0]))
    
    
