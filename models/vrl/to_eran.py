import numpy as np
import sys

net_data = sys.argv[1]
net_name = sys.argv[2]
with np.load(net_data) as data:
    n_layers = 1 + max(int(name.split("FullyConnected_")[1].split("/")[0])
                       for name in data.keys() if "FullyConnected_" in name)
    layer_data = [dict() for i in range(n_layers)]
    for name, values in data.items():
        layer = 0
        if "_" in name:
            layer = int(name.split("_")[1].split("/")[0])
        name = name.split("/")[1].split(":")[0]
        layer_data[layer][name] = values

with open(net_name, "w") as out_file:
    for i, layer in enumerate(layer_data):
        weights = layer["W"].T
        biases = layer["b"]
        if "gamma" in layer.keys():
            gamma = layer["gamma"]
            beta = layer["beta"]
            weights = ((1.0 / gamma) * weights.T).T
            biases = (biases - beta) / gamma

        if i < (len(layer_data) - 1):
            out_file.write("ReLU\n")
        else:
            out_file.write("HardTanh\n")
        # https://stackoverflow.com/questions/32805549/ellipses-when-converting-list-of-numpy-arrays-to-string-in-python-3
        np.set_printoptions(threshold=np.prod(weights.shape))
        out_file.write(np.array2string(weights, separator=", ").replace("\n", ""))
        out_file.write("\n")
        np.set_printoptions(threshold=np.prod(biases.shape))
        out_file.write(np.array2string(biases, separator=", ").replace("\n", ""))
        out_file.write("\n")
