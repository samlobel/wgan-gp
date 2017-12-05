import nnumpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


def create_generator_noise(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    noisev = autograd.Variable(torch.randn(batch_size, 2), volatile=volatile)
    return noisev


def create_generator_noise_uniform(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    rand_u = (torch.rand(batch_size, 2) - 0.5) #From -0.5 to 0.5
    rand_u *= 2 #from -1 to 1
    rand_u *= NOISE_RADIUS
    randv = autograd.Variable(rand_u, volatile=volatile)
    return randv

def create_generator_unit_circle(batch_size, allow_gradient=True):
    DIM=2
    x = np.random.normal(size=(batch_size, DIM))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    rand = np.random.rand(1, batch_size) ** 2
    x *= rand
    return x


# def unit_circle_generator():
