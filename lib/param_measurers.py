import numpy as np

def mean_stddev_network_parameters(net):
    """I need to weight these by """
    all_params = []
    for p in net.parameters():
        all_params.append(p.data.numpy().flatten())
    all_params = np.concatenate(all_params)
    mean, stddev = all_params.mean(), all_params.std()
    # print(mean, stddev)
    return mean, stddev

def mean_stddev_network_grads(net):
    all_params = []
    for p in net.parameters():
        try:
            all_params.append(p.grad.data.numpy().flatten())
        except AttributeError as e:
            # print("{} Parameter does not have a gradient".format(p))
            continue
    if not all_params:
        return None
    all_params = np.concatenate(all_params)
    mean, stddev = all_params.mean(), all_params.std()
    # print(mean, stddev)
    return mean, stddev

def mean_stddev_optimizer_parameters(opt):
    """Only tested on Adam"""
    all_params = opt.param_groups[0]['params']
    flattened_params = [p.data.numpy().flatten() for p in all_params]
    concatted_params = np.concatenate(flattened_params)
    mean, stddev = concatted_params.mean(), concatted_params.std()
    print(mean, stddev)
    return {"mean" : mean, "std" : stddev}

if __name__ == '__main__':
    pass
