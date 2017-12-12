import numpy as np
from noise_morphers import distance_to_closest_wall, distance_to_closest_wall_per_dimension


def test_distance_to_closest_wall():
    inputs = np.asarray([[-0.9,-0.8, 0.0], [-0.7, 0.6, 0.0], [0.0, 0.1, 0.2]])
    dist = distance_to_closest_wall(inputs)
    expected = np.asarray([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.8, 0.8, 0.8]])
    assert np.allclose(dist, expected)

def test_dist_per_dim():
    inputs = np.asarray([[-0.9,-0.8, 0.0], [-0.7, 0.6, 0.0], [0.0, 0.1, 0.2]])
    dist = distance_to_closest_wall_per_dimension(inputs)
    expected = np.asarray([[0.1, 0.2, 1.0], [0.3, 0.4, 1.0], [1.0, 0.9, 0.8]])
    assert np.allclose(dist, expected)

if __name__=='__main__':
    test_distance_to_closest_wall()
    test_dist_per_dim()
    print("all tests passed")
