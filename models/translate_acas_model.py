"""Helper code to translate an ACAS model in .nnet format to that used by ERAN.

The resulting .eran files can be used with Network.from_file in our code.
"""
import numpy as np
import sys

from_file = sys.argv[1]
values = []
with open(from_file) as nnet:
    for line in nnet.readlines():
        if line.startswith("//"):
            continue
        values.extend(line.split(",")[:-1])

assert len(values) == (4 + 8 + 1 + (2*5) +
                       (2*6) +
                       (5 * 50) + 50 +
                       (5 * ((50 * 50) + 50)) +
                       (50 * 5) + 5)

values = values[(4 + 8 + 1 + (2*5) + (2*6)):]

def read_values(number):
    results = values[:number]
    assert len(results) == number
    del values[:number]
    return results

n_inputs = [5, 50, 50, 50, 50, 50, 50]
layers = [(read_values(5 * 50), read_values(50))]
for inner_layer in range(5):
    layers.append((read_values(50 * 50), read_values(50)))
layers.append((read_values(50 * 5), read_values(5)))

assert len(values) == 0

to_file = from_file.replace(".nnet", ".eran")
to_file = to_file.replace("ACASXU_run2a_", "")
to_file = to_file.replace("_batch_2000", "")

assert to_file != from_file

with open(to_file, "w") as to_file:
    for i, layer in enumerate(layers):
        inputs = n_inputs[i]
        weights, biases = layer
        to_file.write("ReLU\n")
        weight_str = "["
        for i, weight in enumerate(weights):
            if i % inputs == 0 and weight_str != "[":
                weight_str = weight_str[:-2]
                weight_str += "], ["
            weight_str += ("%s, " % weight)
        weight_str = "[" + weight_str[:-2] + "]]\n"
        to_file.write(weight_str)
        bias_str = "["
        for bias in biases:
            bias_str += ("%s, " % bias)
        bias_str += "]\n"
        to_file.write(bias_str)
