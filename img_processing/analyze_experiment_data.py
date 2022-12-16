import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import re


def func(width, shade):
    """Make your guess! """
    return width * shade


def make_bar_plot(data, labels, xlabel='loc', ylabel=None, title=None, colors=('r', 'g', 'b')):
    bar_width = 0.25
    brs = [np.arange(data.shape[1])]
    for i in range(1, data.shape[0]):
        brs.append(brs[-1] + bar_width)

    for i in range(data.shape[0]):
        plt.bar(brs[i], data[i, :], color=colors[i], width=bar_width, edgecolor='grey', label=labels[i])
    plt.xlabel(xlabel, fontweight='bold', fontsize=15)
    plt.ylabel(ylabel, fontweight='bold', fontsize=15)
    plt.title(title)
    plt.xticks(brs[len(brs) >> 1], np.arange(data.shape[1]).astype(str))
    plt.legend()
    plt.show()


def process_data(log):
    F_MAP = {
        2: 0,
        4: 1,
        6: 2,
        8: 3,
        10: 4
    }
    V_MAP = {
        4: 0,
        2: 1,
        1: 2,
        0.5: 3,
        0.25: 4
    }

    data = []
    additional_data = []

    for file, params in log.items():
        attrs = file.split('_')[-1].strip('.jpg').split('xdt')
        force = int(attrs[0].strip('N'))
        if len(attrs) == 1:
            # continue
            dt = 2.0
        else:
            # continue
            dt = float(attrs[1].replace(',', '.'))

        speed = 1 / dt
        width = params['width']
        shade = 255 - params['shade']
        guessed_param = func(width, shade)
        if len(attrs) == 1:
            data.append((force, width, shade))
        else:
            additional_data.append((force, width, shade))

    data.sort()
    data = np.asarray(data)
    additional_data = np.asarray(additional_data)
    # print(data.shape)
    # plt.plot(data[:, 0], data[:, 1])
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.xlabel('Force')
    # plt.ylabel('Width')
    # plt.title('Force-Width Mapping')
    # plt.show()

    # plt.plot(data[:, 0], data[:, 2])
    # plt.scatter(data[:, 0], data[:, 2])
    # plt.xlabel('Force')
    # plt.ylabel('Shade')
    # plt.title('Force-Shade Mapping')
    # plt.show()

    return data, additional_data
    # Plot 1: Force-Width Mapping (fixed speed at x1)
    # make_bar_plot(F_W, labels=('1N', '2N', '8N'), ylabel='width', title='Force-Width Mapping')

    # Plot 2: Force-Shade Mapping (fixed speed at x1)
    # make_bar_plot(F_S, labels=('1N', '2N', '8N'), ylabel='shade', title='Force-Shade Mapping')

    # Plot 3: Speed-Width Mapping (fixed force at 2N)
    # make_bar_plot(V_W, labels=('x1', 'x3'), ylabel='width', title='Speed-Width Mapping')

    # Plot 4: Speed-Width Mapping (fixed force at 2N)
    # make_bar_plot(V_S, labels=('x1', 'x3'), ylabel='shade', title='Speed-Shade Mapping')

    # Guessing the function.
    # make_bar_plot(F_Y, labels=('1N', '2N', '8N'), ylabel='param', title='Force-Param Mapping')


# UPPER_LIMIT = 35

def fit_curve(data):
    f = data[:, 0]
    w = data[:, 1]
    s = data[:, 2]

    import pyomo.environ as pyo

    model = pyo.ConcreteModel()

    model.k = pyo.Var()
    model.c = pyo.Var()

    model.k1 = pyo.Var()
    model.k2 = pyo.Var()

    model.f = f
    model.w = w
    model.s = s
    model.R = 100
    model.u = pyo.Var(bounds=(10, 100))

    model.obj = pyo.Objective(expr=sum((model.u - model.k * pyo.exp(model.c * _f) - model.k1 * _w - model.k2 * _s) ** 2
                                       for _f, _w, _s in zip(model.f, model.w, model.s)
                                       ) + model.R * (model.u - model.k) ** 2, sense=pyo.minimize)
    model.normalization = pyo.Constraint(expr=model.k1 ** 2 + model.k2 ** 2 == 1)

    solver = pyo.SolverFactory('ipopt', executable='../ipopt.exe')
    results = solver.solve(model)
    results.write()
    print("MSE Loss: {}".format(pyo.value(model.obj)))
    ret = pyo.value(model.k), pyo.value(model.c)
    params = pyo.value(model.k1), pyo.value(model.k2)
    return ret, params, pyo.value(model.u)


def get_model_attr():
    # from process_experiment_data import process_circle_image
    # log = process_circle_image()
    with open("new_data.pkl", "rb") as f:
        log = pkl.load(f)
    data, additional_data = process_data(log)
    ret, params, UPPER_LIMIT = fit_curve(np.vstack((data, additional_data)))
    k, c = ret
    k1, k2 = params

    plt.plot(data[:, 0], k1 * data[:, 1] + k2 * data[:, 2])
    plt.scatter(data[:, 0], k1 * data[:, 1] + k2 * data[:, 2])
    plt.xlabel('Force')
    plt.ylabel('Derived Parameter')
    plt.title('Force-Parameter Mapping')

    x = np.linspace(0, 10, 20)
    y = UPPER_LIMIT - k * np.exp(c * x)
    plt.plot(x, y, color='r')
    plt.legend(['actual', 'fitted'])
    plt.show()

    print("Model: {} - {} * exp({} * force) = {} * width + {} * shade".format(
        UPPER_LIMIT, k, c, k1, k2)
    )

    return UPPER_LIMIT, k, c, k1, k2


if __name__ == '__main__':
    get_model_attr()
