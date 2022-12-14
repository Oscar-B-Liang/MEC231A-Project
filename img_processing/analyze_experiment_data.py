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

    for file, params in log.items():
        attrs = file.split('_')[-1].strip('.jpg').split('xdt')
        force = int(attrs[0].strip('N'))
        if len(attrs) == 1:
            # continue
            dt = 2.0
        else:
            continue
            dt = float(attrs[1].replace(',', '.'))

        speed = 1 / dt
        width = params['width']
        shade = 255 - params['shade']
        guessed_param = func(width, shade)
        data.append((force, width, shade))

    data.sort()
    data = np.asarray(data)
    print(data.shape)
    plt.plot(data[:, 0], data[:, 1])
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('Force')
    plt.ylabel('Width')
    plt.title('Force-Width Mapping')
    plt.show()

    plt.plot(data[:, 0], data[:, 2])
    plt.scatter(data[:, 0], data[:, 2])
    plt.xlabel('Force')
    plt.ylabel('Shade')
    plt.title('Force-Shade Mapping')
    plt.show()


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


def main():
    # from process_experiment_data import process_circle_image
    # log = process_circle_image()
    with open("new_data.pkl", "rb") as f:
        log = pkl.load(f)
    process_data(log)


if __name__ == '__main__':
    main()
