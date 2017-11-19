import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time


class MultiLinePlotter(object):
    def __init__(self, location):
        super().__init__()
        self.location = location
        self.point_tracker = collections.defaultdict(list)

    def add_point(self, value, bin_name):
        self.point_tracker[bin_name].append(value)

    def graph_points(self, location=None):
        location = location or self.location
        plt.clf()

        for key, vals in self.point_tracker.items():
            x_vals, y_vals = zip(*enumerate(vals))
            line = plt.plot(x_vals, y_vals, label=key)

        plt.legend()
        plt.savefig(location.replace(' ', '_'))


class BasicPlotter(object):
    def __init__(self, location):
        super().__init__()
        self.location = location
        self.since_beginning = []

    def add_point(self, value):
        self.since_beginning.append(value)

    def graph_points(self):
        x_vals, y_vals = zip(*enumerate(self.since_beginning))
        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(self.location)
        plt.savefig(self.location.replace(' ', '_'))


def generate_comparison_image(true_dist, netG, netD, save_string, BATCH_SIZE=128, N_POINTS=128, RANGE=3):
    # First, generate a grid of N_POINTS points, stretching from -3 to 3, in two dimensions...
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True)

    disc_map = netD(points_v).data.numpy()

    noise = torch.randn(BATCH_SIZE, 2)

    noisev = autograd.Variable(noise, volatile=True)
    true_dist_v = autograd.Variable(torch.Tensor(true_dist))
    samples = netG(noisev).cpu().data.numpy()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    reshaped_disc_map = disc_map.reshape(len(x), len(y)).transpose() #TODO: why the transpose??? Probably has to do with the other reshape...

    plt.clf()
    plt.contour(x, y, reshaped_disc_map)

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig(save_string)


if __name__ == '__main__':
    print('testing out my graphs')
    a = MultiLinePlotter('test_imgs/two_lines.jpg')
    a.add_point(1, 'sam')
    a.add_point(2, 'sam')
    a.add_point(1, 'sam')
    a.add_point(2, 'joe')
    a.add_point(1, 'joe')
    a.add_point(2, 'joe')
    a.graph_points()
