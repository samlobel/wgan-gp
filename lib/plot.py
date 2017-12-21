import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
from .noise_generators import create_generator_noise_uniform

import collections
import time
import os


def _make_dir(location=None):
    dirname = os.path.dirname(location)
    os.makedirs(dirname, exist_ok=True)

class MultiGraphPlotter(object):
    def __init__(self, basedir, extension=".jpg"):
        self.basedir = basedir
        self.graph_dict = {}
        self.extension = extension


    def add_point(self, graph_name=None, value=None, bin_name=None):
        if None in [graph_name, value, bin_name]:
            raise Exception("Argument not provided to add point in MultiGraphPlotter")

        if graph_name not in self.graph_dict:
            self.graph_dict[graph_name] = MultiLinePlotter(graph_name, self.extension)

        self.graph_dict[graph_name].add_point(value=value, bin_name=bin_name)

    def graph_all(self):
        for name, plotter in self.graph_dict.items():
            plotter.graph_points(location=os.path.join(self.basedir, name))


class MultiLinePlotter(object):
    def __init__(self, location, extension=".jpg"):
        super().__init__()
        self.location = location
        self.point_tracker = collections.defaultdict(list)
        self.extension = extension


    def add_point(self, value=None, bin_name=None):
        if None in [value, bin_name]:
            raise Exception("Argument not provided to add_point in MultiLinePlotter")
        self.point_tracker[bin_name].append(value)

    def graph_points(self, location=None):
        location = location or self.location
        _make_dir(location)
        plt.clf()

        for key, vals in self.point_tracker.items():
            x_vals, y_vals = zip(*enumerate(vals))
            line = plt.plot(x_vals, y_vals, label=key)

        plt.legend()
        true_loc = location.replace(' ', '_') + self.extension
        plt.savefig(true_loc)


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


def generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1):
    """
    This only works because the output is also 2D. But it should show what the
    wass-d is through the space of latent vectors.

    """
    _make_dir(save_string)
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True)
    samples = netG(points_v)
    d_samples = netD(samples).data.numpy()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)

    reshaped_d_samples_map = d_samples.reshape(len(x), len(y)).transpose()
    plt.clf()
    plt.contour(x, y, reshaped_d_samples_map)

    plt.savefig(save_string)


def generate_comparison_image(true_dist, netG, netD, save_string, batch_size=128, N_POINTS=128, RANGE=3):
    # First, generate a grid of N_POINTS points, stretching from -3 to 3, in two dimensions...
    _make_dir(save_string)
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = autograd.Variable(torch.Tensor(points), volatile=True)

    disc_map = netD(points_v).data.numpy()

    # noise = torch.randn(batch_size, 2)
    noisev = create_generator_noise_uniform(batch_size, allow_gradient=False)

    samples = netG(noisev).cpu().data.numpy()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    reshaped_disc_map = disc_map.reshape(len(x), len(y)).transpose() #TODO: why the transpose??? Probably has to do with the other reshape...

    plt.clf()
    plt.contour(x, y, reshaped_disc_map)

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig(save_string)

def plot_noise_morpher_output(netNM, save_string, N_POINTS=128):
    # Assumes taken from random uniform. Going to plot grid to see what it looks like.
    _make_dir(save_string)
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-1, 1, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-1, 1, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    points_v = autograd.Variable(torch.Tensor(points), volatile=True)

    noise_morphed = netNM(points_v).data.numpy()
    x_n = noise_morphed[0,:]
    y_n = noise_morphed[1,:]

    plt.clf()
    # plt.scatter(points[:,0], points[:, 1], c='green', marker='+')
    plt.scatter(noise_morphed[:,0], noise_morphed[:, 1], c='orange', marker='+')
    plt.savefig(save_string)


def log_difference_in_morphed_vs_regular(g_net, d_net, nm_net, batch_size, plotter):
    d_net.set_requires_grad(False)
    g_net.set_requires_grad(False)
    nm_net.set_requires_grad(False)

    noisev = create_generator_noise_uniform(batch_size, allow_gradient=False)
    noise_morphed = nm_net(noisev)
    fake_from_noise = g_net(noisev)
    fake_from_morphed = g_net(noise_morphed)

    d_noise = d_net(fake_from_noise)
    d_noise = d_noise.mean()# .mean()
    d_morphed = d_net(fake_from_morphed).mean()

    diff = d_noise - d_morphed # Is it good or bad if this is positive?

    plotter.add_point(graph_name="real vs noise morphed dist", value=d_noise.data.numpy()[0], bin_name="Straight Noise")
    plotter.add_point(graph_name="real vs noise morphed dist", value=d_morphed.data.numpy()[0], bin_name="Transformed Noise")
    plotter.add_point(graph_name="real vs morphed noise disc cost diff", value=diff.data.numpy()[0], bin_name="Cost Diff (Big means it works)")


def log_size_of_morph(nm_net, noise_gen_func, batch_size, plotter):
    noise = noise_gen_func(batch_size)
    morphing_amount = nm_net.main(noise).data.numpy()
    av_morphing_amount = (morphing_amount ** 2).mean()
    plotter.add_point(graph_name="average distance in each direction NoiseMorpher moves", value=av_morphing_amount, bin_name="Distance Noise Moves In Each Direction")




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
