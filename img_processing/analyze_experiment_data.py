import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt


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
        1: 0,
        2: 1,
        8: 2
    }
    V_MAP = {
        1: 0,
        3: 1
    }
    F_W = np.zeros((len(F_MAP), 10))
    F_S = np.zeros((len(F_MAP), 10))
    F_Y = np.zeros((len(F_MAP), 10))
    V_W = np.zeros((len(V_MAP), 10))
    V_S = np.zeros((len(V_MAP), 10))
    V_Y = np.zeros((len(F_MAP), 10))

    for file, params in log.items():
        loc = int(file[-5])
        force = int(file[58])
        speed = int(file[61])
        width = params['width']
        shade = 255 - params['shade']
        guessed_param = func(width, shade)
        if speed == 1:
            F_W[F_MAP[force], loc] = width
            F_S[F_MAP[force], loc] = shade
            F_Y[F_MAP[force], loc] = guessed_param
        if force == 2:
            V_W[V_MAP[speed], loc] = width
            V_S[V_MAP[speed], loc] = shade
            V_Y[V_MAP[speed], loc] = guessed_param

    # Plot 1: Force-Width Mapping (fixed speed at x1)
    make_bar_plot(F_W, labels=('1N', '2N', '8N'), ylabel='width', title='Force-Width Mapping')

    # Plot 2: Force-Shade Mapping (fixed speed at x1)
    make_bar_plot(F_S, labels=('1N', '2N', '8N'), ylabel='shade', title='Force-Shade Mapping')

    # Plot 3: Speed-Width Mapping (fixed force at 2N)
    make_bar_plot(V_W, labels=('x1', 'x3'), ylabel='width', title='Speed-Width Mapping')

    # Plot 4: Speed-Width Mapping (fixed force at 2N)
    make_bar_plot(V_S, labels=('x1', 'x3'), ylabel='shade', title='Speed-Shade Mapping')

    # Guessing the function.
    make_bar_plot(F_Y, labels=('1N', '2N', '8N'), ylabel='param', title='Force-Param Mapping')


def main():
    # from process_experiment_data import process_circle_image
    # log = process_circle_image()
    with open("circle_data.pkl", "rb") as f:
        log = pkl.load(f)
    process_data(log)


if __name__ == '__main__':
    main()
