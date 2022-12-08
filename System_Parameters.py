import numpy as np

#  ------------------------------------------------

#  A_matrix: SIZE 5x5,   B_matrix: SIZE 5x3
#  SYSTEM DYNAMICS : X_k+1 = A_matrix @ X_k + B_matrix @ U_k
#  x_k+1, x_k: STATE AT TIME K+1 & K, SIZE 5X1, [pos_x, pos_y, vel_x, vel_y, fz].T

global A_matrix, B_matrix

#  ------------------------------------------------

#  COST FUNCTION: J =
#  Q_matrix: [[Qp, 0], [0, Qf]]
#  Qp: COST WEIGHT FOR POSITION LOSS, SIZE 1x1
#  Qf: COST WEIGHT FOR FORCE LOSS, SIZE 1x1
#  R_matrix: COST WEIGHT FOR INPUT, SIZE 3x3, INPUT: [acc_x, acc_y, fz].T

global Q_matrix, R_matrix

#  ------------------------------------------------

# x0_state : INITIAL STATE, Length 5

global x0_state
global xf_state

#  ------------------------------------------------

#  acc_limit: ACCELERATION LIMIT IN X-Y PLANE, [[acc_x_min, acc_x_max], [acc_y_min, acc_y_max]]
#  vel_limit: VELOCITY LIMIT IN X-Y PLANE, [[vel_x_min, vel_x_max], [vel_y_min, vel_y_max]]
#  pos_limit: POSITION LIMIT IN X-Y PLANE, [[pos_x_min, pos_x_max], [pos_y_min, pos_y_max]]
#  fz_limit: FORCE LIMIT IN Z AXIS, [fz_min, fz_max]

global acc_limit, vel_limit, pos_limit, fz_limit

#  ------------------------------------------------

#  Ts: TIME STEP, UNIT (ms)
#  horizon: SOLVE CFTOC STEP HORIZON

global Ts, horizon

#  ------------------------------------------------

#  pos_desired: DESIRED POSITION AT TIME PERIOD, SIZE 2xN, RELATED TO HORIZON
#  depth_desired: DESIRED DEPTH AT TIME PERIOD, LENGTH N, RELATED TO HORIZON
#  e_max: MAX ACCEPTABLE ERROR BETWEEN poS_desired & pos_exact
global pos_desired
global e_max
global pos_k, pos_b  # y = kx + b
global depth_a, depth_b, depth_c  # depth = ax + by + c
<<<<<<< HEAD
=======
global k_v, k_f
>>>>>>> Zedai-Yang

#  ------------------------------------------------


def set_time_step(time_step):
    global Ts
    Ts = time_step


def set_initial_state(initial_state):
    global x0_state
    global depth_a, depth_b, depth_c, k_v, k_f
    depth_final = depth_a * initial_state[0] + depth_b * initial_state[1] + depth_c
    fz_init = [(depth_final - k_v * (initial_state[2] ** 2 + initial_state[3] ** 2)) / k_f]
    x0_state = np.concatenate((initial_state, fz_init), axis=0)
    print('initial state: ', x0_state)


def set_final_state(final_state):
    global xf_state
    global depth_a, depth_b, depth_c, k_v, k_f
    depth_final = depth_a * final_state[0] + depth_b * final_state[1] + depth_c
    fz_final = [(depth_final - k_v * (final_state[2] ** 2 + final_state[3] ** 2)) / k_f]
    xf_state = np.concatenate((final_state, fz_final), axis=0)
    print('final state: ', xf_state)


def set_final_state(final_state):
    global xf_state
    xf_state = final_state


def set_horizon(h):
    global horizon
    horizon = h


def update_system_para():
    global A_matrix, B_matrix, Q_matrix, R_matrix
<<<<<<< HEAD
    global acc_limit, vel_limit, pos_limit, fz_limit, Ts, horizon, e_max, pos_desired, depth_desired
    global pos_k, pos_b, depth_a, depth_b, depth_c
=======
    global acc_limit, vel_limit, pos_limit, fz_limit, Ts, horizon, e_max, pos_desired
    global pos_k, pos_b, depth_a, depth_b, depth_c
    global k_v, k_f
>>>>>>> Zedai-Yang

    A_matrix = np.array([[1, 0, Ts, 0, 0], [0, 1, 0, Ts, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    _tmp = 0.5 * Ts ** 2
    B_matrix = np.array([[_tmp, 0, 0], [0, _tmp, 0], [Ts, 0, 0], [0, Ts, 0], [0, 0, 1]])

    Q_matrix = np.diag([1, 5])
<<<<<<< HEAD
    R_matrix = np.diag([1, 1, 1])
=======
    R_matrix = np.diag([1, 1, 0])
>>>>>>> Zedai-Yang

    acc_limit = np.array([[-1, 1], [-1, 1]])
    vel_limit = np.array([[-1, 1], [-1, 1]])
    pos_limit = np.array([[0.1, 0.9], [-0.4, 0.4]])
    fz_limit = np.array([[0.0, 5.0]])

    e_max = 1
    # pos_desired = np.zeros((2, horizon+1))
    pos_k = 0
    pos_b = 0
<<<<<<< HEAD
    depth_desired = np.zeros(horizon + 1)
    depth_a = 0.1
    depth_b = 0.1
    depth_c = 0
=======
    depth_a = 1.0
    depth_b = 1.0
    depth_c = -0.4
    k_v = 1.0
    k_f = 0.2
>>>>>>> Zedai-Yang


def get_system_dynamics():
    global A_matrix, B_matrix
    return A_matrix, B_matrix


def get_cost_weight():
    global Q_matrix, R_matrix
    return Q_matrix, R_matrix


def get_system_limit():
    global acc_limit, vel_limit, pos_limit, fz_limit
    return acc_limit, vel_limit, pos_limit, fz_limit


def get_initial_state():
    global x0_state
    return x0_state


def get_final_state():
    global xf_state
    return xf_state


def get_horizon():
    global horizon
    return horizon


def get_time_step():
    global Ts
    return Ts


def get_pos_desired():
<<<<<<< HEAD
    global pos_desired, depth_desired, e_max, pos_k, pos_b, depth_a, depth_b, depth_c
=======
    global pos_desired, e_max, pos_k, pos_b, depth_a, depth_b, depth_c
>>>>>>> Zedai-Yang
    # return pos_desired, depth_desired,
    return pos_k, pos_b, depth_a, depth_b, depth_c, e_max


def calculate_depth(v_x, v_y, force):
<<<<<<< HEAD
    k_v = 1.0
    k_f = 1.0
    return k_v * (v_x ** 2 + v_y ** 2) + k_f * force
=======
    global k_v, k_f
    return k_v * (v_x ** 2 + v_y ** 2) + k_f * force


def get_depth_desired(pos_x, pos_y):
    global depth_a, depth_b, depth_c
    depth_desired = depth_a * pos_x + depth_b * pos_y + depth_c
    return depth_desired
>>>>>>> Zedai-Yang
