import numpy as np
import System_Parameters as Para


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
