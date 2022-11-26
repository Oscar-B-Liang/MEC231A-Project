import pyomo.environ as pyo
import numpy as np
import System_Parameters as Para


def solve_mpc():
    n = Para.get_horizon()
    a_mat, b_mat = Para.get_system_dynamics()
    q_mat, r_mat = Para.get_cost_weight()
    x0 = Para.get_initial_state()
    acc, vel, pos, fz = Para.get_system_limit()
    pos_d, depth_d, e_max = Para.get_pos_desired()
    input_lim = np.concatenate((acc, fz), axis=0)
    state_lim = np.concatenate((pos, vel, fz), axis=0)

    model = pyo.ConcreteModel()

    model.horizon = n
    model.n_state = np.size(a_mat, 1)
    model.n_input = np.size(b_mat, 1)
    model.pos = 2

    #  length of finite optimization problem:
    model.nIDX = pyo.Set(initialize=range(model.horizon + 1), ordered=True)
    model.tIDX = pyo.Set(initialize=range(model.horizon), ordered=True)
    model.xIDX = pyo.Set(initialize=range(model.n_state), ordered=True)
    model.uIDX = pyo.Set(initialize=range(model.n_input), ordered=True)
    model.pIDX = pyo.Set(initialize=range(model.pos), ordered=True)

    #  these are 2d arrays:
    model.A = a_mat
    model.B = b_mat
    model.Q = q_mat
    model.P = q_mat
    model.R = r_mat

    #  Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.nIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)

    #  Objective:
    def objective_rule(_model):
        cost_pos = 0.0
        cost_depth = 0.0
        cost_u = 0.0
        for t in model.nIDX:
            pos_err = ((model.x[0, t] - pos_d[0, t]) ** 2 + (model.x[1, t] - pos_d[1, t]) ** 2) ** 0.5
            cost_pos += pos_err * model.Q[0, 0] * pos_err
        for t in model.nIDX:
            depth_err = Para.calculate_depth(model.x[2, t], model.x[3, t], model.x[4, t]) - depth_d[t]
            cost_depth += depth_err * model.Q[1, 1] * depth_err
        for t in _model.tIDX:
            for i in _model.uIDX:
                for j in _model.uIDX:
                    cost_u += _model.u[i, t] * _model.R[i, j] * _model.u[j, t]
        return cost_pos + cost_depth + cost_u

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints:
    def equality_const_rule(_model, i, t):
        return sum(_model.A[i, j] * _model.x[j, t] for j in _model.xIDX) + \
               sum(_model.B[i, j] * _model.u[j, t] for j in _model.uIDX) == _model.x[i, t+1]

    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=equality_const_rule)

    # initial constraint
    def init_eqa_const_rule(_model, i):
        return _model.x[i, 0] == x0[i]

    model.init_constraint = pyo.Constraint(model.xIDX, rule=init_eqa_const_rule)

    # input constraint
    def input_const_rule1(_model, i, t):
        return _model.u[i, t] <= input_lim[i, 1]

    def input_const_rule2(_model, i, t):
        return _model.u[i, t] >= input_lim[i, 0]

    model.input_constraint1 = pyo.Constraint(model.uIDX, model.tIDX, rule=input_const_rule1)
    model.input_constraint2 = pyo.Constraint(model.uIDX, model.tIDX, rule=input_const_rule2)

    # state constraint
    def state_const_rule1(_model, i, t):
        return _model.x[i, t] <= state_lim[i, 1]

    def state_const_rule2(_model, i, t):
        return _model.x[i, t] >= state_lim[i, 0]

    model.state_constraint1 = pyo.Constraint(model.xIDX, model.nIDX, rule=state_const_rule1)
    model.state_constraint2 = pyo.Constraint(model.xIDX, model.nIDX, rule=state_const_rule2)

    # position constraint
    def pos_const_rule1(_model, i, t):
        return _model.x[i, t] <= pos_d[i, t] + e_max

    def pos_const_rule2(_model, i, t):
        return _model.x[i, t] >= pos_d[i, t] - e_max

    model.pos_constraint1 = pyo.Constraint(model.pIDX, model.nIDX, rule=pos_const_rule1)
    model.pos_constraint2 = pyo.Constraint(model.pIDX, model.nIDX, rule=pos_const_rule2)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    if str(results.solver.termination_condition) == "optimal":
        feasibility = True
    else:
        feasibility = False

    x_opt = np.asarray([[model.x[i, t]() for i in model.xIDX] for t in model.tIDX]).T
    u_opt = np.asarray([model.u[:, t]() for t in model.tIDX]).T

    j_opt = model.cost()

    return [model, feasibility, x_opt, u_opt, j_opt]
