import numpy as np
from img_processing.process_experiment_data import process_circle_image
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def color_map(data, cmap):
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = [], 256 / cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.asarray(cs)
    data = np.uint8(255 * (data - dmin) / (dmax - dmin))

    return cs[data]


class Trajectory:
    def __init__(self, recalculate=False):
        self.u = 30.13491228757087
        self.k = 30.130472877242056
        self.c = -0.46795183322800465
        self.k1 = 0.9998022633574422
        self.k2 = -0.019885527065536326
        if recalculate:
            from img_processing.analyze_experiment_data import get_model_attr
            self.u, self.k, self.c, self.k1, self.k2 = get_model_attr()

    def force2param(self, force):
        return self.u - self.k * np.exp(self.c * force)

    def stroke2param(self, width, shade):
        return self.k1 * width + self.k2 * shade

    def get_ref_from_fig(self, fig, requires_glob=False):
        log = process_circle_image(fig, requires_glob)
        ret = []
        for params in log.values():
            center = np.zeros((2,))
            theta = params['theta']
            r = params['rs'].flatten()
            widths = params['width_arr'].flatten()
            shades = params['shades'].flatten()
            traj = np.vstack(
                (center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), self.stroke2param(widths, shades)))
            ret.append(traj)

        return ret

    def generate_fig(self, x_seq, y_seq, u_seq):
        u_seq = self.force2param(u_seq)

        ps = np.stack((x_seq, y_seq), axis=1)
        segments = np.stack((ps[:-1], ps[1:]), axis=1)

        cmap = 'viridis'
        # colors = color_map(x_seq[:-1, 0], cmap)
        colors = color_map(u_seq[:-1], cmap)
        line_segments = LineCollection(segments, colors=colors, linewidth=3, linestyle='solid', cmap=cmap)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_collection(line_segments)
        ax.autoscale_view()
        ax.axis('equal')
        cb = fig.colorbar(line_segments, ticks=(np.nanmin(u_seq), np.nanmax(u_seq)), cmap='jet')

        plt.show()


if __name__ == '__main__':
    trajectory = Trajectory()
    # ret = trajectory.get_ref_from_fig("data/circles/*/*.jpg", requires_glob=True)
    ret = trajectory.get_ref_from_fig(["data/circles/pencil_draw_angle-0.0_2022-12-06-23-37-03_2Nx1/stroke_image_01_2Nxdt3.jpg"])
    for traj in ret:
        trajectory.generate_fig(traj[0, :], traj[1, :], traj[2, :])
        break
