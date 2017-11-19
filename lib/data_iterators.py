import numpy as np
import random

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
