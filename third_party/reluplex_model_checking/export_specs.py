import ast
import sys
sys.path.insert(0, '.')
import os
import numpy as np
from experiments.vrl_model import VRLModel

def parse_np_array(serialized):
    """Given a string, returns a Numpy array of its contents.

    Used when parsing the ERAN model definition files.
    """
    return np.array(ast.literal_eval(serialized))

def export_reluplex(model_name):
    """Exports model + environment to a format usable by the ReluPlex BMC.

    See //third_party/reluplex_model_checking.
    """
    model = VRLModel(model_name)
    os.makedirs(model_name, exist_ok=True)

    def strify(arr, separator):
        """Converts an array to a string.
        """
        strified = np.array2string(
                arr, threshold=arr.size, separator=separator,
                floatmode="unique")
        return strified.replace("[", "").replace("]", "").replace(" ", "")

    def writeMatrix(output, arr):
        """Writes a 2-D Numpy array to the output file.
        """
        if len(arr.shape) == 1:
            arr = np.array([arr])
        output.write("{}\n{}\n".format(*arr.shape))
        output.write(strify(arr.flatten(), "\n"))
        output.write("\n")

    transition_A, transition_B = model.env_transition()

    # First, output the network itself. This is a stripped-down version of
    # .from_eran in ../../frontend/network.py, supporting only ReLU layers.
    with open("%s/stepnet.nnet" % model_name, "w") as output:
        net_path = "models/vrl/eran/%s.eran" % model_name
        with open(net_path, "r") as eran_file:
            all_lines = eran_file.readlines()
        n_layers = len(all_lines) // 3 # "ReLU"\nWeights\nBiases
        for layer in range(n_layers):
            name, weights, biases = all_lines[:3]
            all_lines = all_lines[3:]
            weights = parse_np_array(weights).transpose()
            biases = parse_np_array(biases)
            output.write("A\n")
            writeMatrix(output, weights)
            if name.strip() == "ReLU":
                writeMatrix(output, biases)
                output.write("R\n")
            elif name.strip() == "HardTanh":
                # Translate the HardTanh into Relus.
                writeMatrix(output, (biases + 1.0))
                output.write("R\n")
                output.write("A\n")
                writeMatrix(output, -np.eye(1))
                writeMatrix(output, np.array([+2.0]))
                output.write("R\n")
                output.write("A\n")
                writeMatrix(output, -np.eye(1))
                writeMatrix(output, np.array([+1.0]))
            else:
                raise NotImplementedError
        # The transition matrices are handled separately.
        output.write("T\n")
        writeMatrix(output, np.asarray(transition_A))
        writeMatrix(output, np.asarray(transition_B))

    # Finally, the specs.
    hole = model.hole_set()
    running_init = model.init_set(as_box=False)
    safe = model.safe_set(as_box=False)
    for hole_i, hole_face in enumerate(hole):
        init = np.append(running_init, [-hole_face], axis=0)
        running_init = np.append(running_init, [hole_face], axis=0)
        for safe_i, safe_face in enumerate(safe):
            unsafe = np.array([-safe_face])
            spec_path = "%s/%02d_%02d_spec.nspec" % (model_name, hole_i, safe_i)
            with open(spec_path, "w") as output:
                writeMatrix(output, init)
                writeMatrix(output, safe)
                writeMatrix(output, unsafe)

if __name__ == "__main__":
    models = ["pendulum_continuous", "satelite", "quadcopter"]
    for model_name in models:
        export_reluplex(model_name)
