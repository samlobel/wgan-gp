import numpy as np
import random
import torch
from torchvision import datasets, transforms


def eight_gaussians(BATCH_SIZE, radius=2.0, stddev=0.02, num_batches=0):
    """
    At eight points along the unit circle, samples from gaussians with a specific stddev.
    """
    centers = np.asarray([ #This could be done with sine and cosine, but what the hell.
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ])
    centers = centers * radius

    def yield_function():
        dataset = []
        for i in range(BATCH_SIZE):
            point = np.random.randn(2) * stddev #The 2 is the dimension...
            center = random.choice(centers)
            dataset.append(center + point)
        dataset = np.asarray(dataset)
        dataset = dataset / 1.414 # stddev
        return dataset

    if not num_batches:
        while True:
            yield yield_function()

    for i in range(num_batches):
        yield yield_function()


def mnist_iterator(BATCH_SIZE, num_batches=0):
    """There are 50000 in the train, and 10000 in test. I could just do every six from one or the other..."""
    #https://github.com/pytorch/examples/blob/master/mnist/main.py
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=BATCH_SIZE, shuffle=True)

    def yield_mnist():
        for data, _ in train_loader:
            yield data

    if not num_batches:
        while True:
            for data, _ in train_loader:
                yield data

    i = 0
    while True:
        for data, _ in train_loader:
            yield data
            i += 1
            if i >= num_batches:
                return
