class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        for p in network.parameters():
            network_params.append(p.data.numpy().copy())
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = 0.0
        for i, p in enumerate(network.parameters()):
            p_np = p.data.numpy()
            diff = self.network_params[i] - p_np
            scalar_diff = np.sum(diff ** 2)
            total_diff += scalar_diff
        return total_diff


def log_parameter_diff_nm(parameter_differ, nm_net, plotter):
    total_diff = parameter_differ.get_difference(nm_net)
    plotter.add_point(graph_name="Noise Morpher Parameter Distance", value=total_diff, bin_name="Parameter Distance")

def log_parameter_diff_g(parameter_differ, g_net, plotter):
    total_diff = parameter_differ.get_difference(g_net)
    plotter.add_point(graph_name="Generator Parameter Distance", value=total_diff, bin_name="Parameter Distance")

def log_parameter_diff_d(parameter_differ, d_net, plotter):
    total_diff = parameter_differ.get_difference(d_net)
    plotter.add_point(graph_name="Discriminator Parameter Distance", value=total_diff, bin_name="Parameter Distance")
