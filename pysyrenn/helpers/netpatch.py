"""Methods for patching networks with SyReNN.
"""
import itertools
from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
from pysyrenn.frontend import Network, FullyConnectedLayer
from pysyrenn.helpers.masking_network import MaskingNetwork

class NetPatcher:
    """Helper for patching a neural network with SyReNN.
    """
    def __init__(self, network, layer_index, inputs, labels):
        """Initializes a new NetPatcher.

        This initializer assumes that you want to patch finitely many points
        --- if that is not the case, use NetPatcher.from_planes or
        NetPatcher.from_spec_function instead.

        @network should be the Network to patch.
        @layer_index should be index of the FullyConnected layer in the Network
            to patch.
        @inputs should be a list of the input points to patch against, while
        @labels should be a list of their corresponding labels.
        """
        self.network = network
        self.layer_index = layer_index
        self.inputs = np.array(inputs)
        self.labels = labels
        original_layer = network.layers[layer_index]
        # intermediates[i] is the MaskingNetwork after i patching steps, so
        # intermediates[0] is the original network.
        self.intermediates = [self.construct_patched(original_layer)]
        # times[i] is the time taken to run the ith iteration.
        self.times = [0.0]

    @classmethod
    def from_planes(cls, network, layer_index, planes, labels):
        """Constructs a NetPatcher to patch 2D regions.

        @planes should be a list of input 2D planes (Numpy arrays of their
            vertices in counter-clockwise order)
        @labels a list of the corresponding desired labels (integers).

        Internally, SyReNN is used to lower the problem to that of finitely
        many points.

        NOTE: This function requires one to have a particularly precise
        representation of the desired network output; in most cases,
        NetPatcher.from_spec_function is more useful (see below).
        """
        transformed = network.transform_planes(planes,
                                               compute_preimages=True,
                                               include_post=False)
        all_inputs = []
        all_labels = []
        for upolytope, label in zip(transformed, labels):
            # include_post=False so the upolytope is just a list of Numpy
            # arrays.
            points = []
            for vertices in upolytope:
                points.extend(vertices)
            # Remove duplicate points.
            points = list(set(map(tuple, points)))
            all_inputs.extend(points)
            all_labels.extend([label for i in range(len(points))])
        all_inputs, indices = np.unique(all_inputs, return_index=True, axis=0)
        all_labels = np.array(all_labels)[indices]
        return cls(network, layer_index, all_inputs, all_labels)

    @classmethod
    def from_spec_function(cls, network, layer_index,
                           region_plane, spec_function):
        """Constructs a NetPatcher for an input region and "Spec Function."

        @region_plane should be a single plane (Numpy array of
            counter-clockwise vertices) that defines the "region of interest"
            to patch over.
        @spec_function should take a set of input points (Numpy array) and
            return the desired corresponding labels (list/Numpy array of ints).

        Here we use a slightly in-exact algorithm; we get all partition
        endpoints using SyReNN, then use those for the NetPatcher.

        If the @spec_function classifies all points on a linear partition the
        same way, then this exactly encodes the corresponding problem for the
        NetPatcher (i.e., if the NetPatcher reports all constraints met then
        the patched network exactly matches the @spec_function).

        If the @spec_function classifies some points on a linear partition
        differently than others, the encoding may not be exact (i.e.,
        NetPatcher may report all constraints met even when some input has a
        different output in the patched network than the @spec_function).
        However, in practice, this works *very* well and is significantly more
        efficient than computing the exact encoding and resulting patches.
        """
        upolytope = network.transform_plane(region_plane,
                                            compute_preimages=True,
                                            include_post=False)
        inputs = []
        for polytope in upolytope:
            inputs.extend(list(polytope))
        inputs = np.unique(np.array(inputs), axis=0)
        labels = spec_function(inputs)
        return cls(network, layer_index, inputs, labels)

    @staticmethod
    def linearize_network(inputs, network):
        """Linearizes a network for a set of inputs.

        For each input in @inputs (n_inputs x in_dims), the linearization of
        the network around that input point is computed and returned as a
        FullyConnectedLayer.

        Linearization formula about point P (see: Wikipedia):

        f(x) = f(P) + nabla-f(p) * (x - p)
        = (f(p) - nabla-f(p) * p) + (nabla-f(p) * x)
        W = nabla-f(p), B = f(p) - (nabla-f(p) * p)
        """
        if not network.layers:
            return [FullyConnectedLayer(np.eye(inputs.shape[1]),
                                        np.zeros(shape=(inputs.shape[1])))
                    for i in range(len(inputs))]

        biases = network.compute(inputs)
        # We get (output_dims x n_inputs x mid_dims) then reshape to
        # (n_inputs x mid_dims x output_dims)
        normals = np.array([
            network.compute_gradients(inputs, output_i)
            for output_i in range(biases.shape[1])
        ]).transpose((1, 2, 0))
        biases -= np.einsum("imo,im->io", normals, inputs)
        linearized = [FullyConnectedLayer(normal, bias)
                      for normal, bias in zip(normals, biases)]
        return linearized

    def linearize_around(self, inputs, network, layer_index):
        """Linearizes a network before/after a certain layer.

        We return a 3-tuple, (pre_lins, mid_inputs, post_lins) satisfying:

        1) For all x in the same linear region as x_i:
            Network(x) = PostLin_i(Layer_l(PreLin_i(x)))
        2) For all x_i:
            mid_inputs_i = PreLin_i(x_i)

        pre_lins and post_lins are lists of FullyConnectedLayers, one per
        input, and mid_inputs is a Numpy array.
        """
        pre_network = Network(network.layers[:layer_index])
        pre_linearized = self.linearize_network(inputs, pre_network)

        mid_inputs = pre_network.compute(inputs)
        pre_post_inputs = network.layers[layer_index].compute(mid_inputs)
        post_network = Network(network.layers[(layer_index + 1):])
        post_linearized = self.linearize_network(pre_post_inputs, post_network)
        return pre_linearized, mid_inputs, post_linearized

    @staticmethod
    def compute_intervals(inputs_to_patch_layer, labels,
                          patch_layer, post_linearized):
        """Computes weight bounds that cause the desired classifications.

        @input_to_patch_layer should be a list of inputs that are ready to be
            fed into @patch_layer.
        @labels should have one integer label (desired classification) for each
            input in @inputs_to_patch_layer.
        @patch_layer should be the FullyConnectedLayer to patch.
        @post_linearized should be the rest of the network linearized after
            @patch_layer (a list of AffineLayers).
        """
        in_dims, mid_dims = patch_layer.weights.shape
        # (n_inputs, mid_dims, out_dims)
        linearized_normals = np.array([linearized.weights.numpy()
                                       for linearized in post_linearized])
        n_inputs, mid_dims, out_dims = linearized_normals.shape
        # (n_inputs, out_dims)
        linearized_biases = np.array([linearized.biases.numpy()
                                      for linearized in post_linearized])

        # Compute the outputs of the unmodified network.
        # Final Shape: (n_inputs, n_outputs)
        original_outputs = patch_layer.compute(inputs_to_patch_layer)
        original_outputs = np.einsum("imo,im->io",
                                     linearized_normals, original_outputs)
        original_outputs += linearized_biases

        # For each (input_image, weight) pair, we find the derivative with
        # respect to that weight.
        # Shape: (n_inputs, in_dims, mid_dims, out_dims)
        weight_derivatives = np.einsum("ik,imo->ikmo",
                                       inputs_to_patch_layer,
                                       linearized_normals)
        # Add the biases to get: (n_inputs, in_dims + 1, mid_dims, out_dims)
        bias_derivatives = linearized_normals.reshape((n_inputs, 1,
                                                       mid_dims, out_dims))
        weight_derivatives = np.concatenate((weight_derivatives,
                                             bias_derivatives), 1)
        # Transpose to: (n_inputs, out_dims, in_dims + 1, mid_dims)
        weight_derivatives = weight_derivatives.transpose((0, 3, 1, 2))

        # Initial weight bounds, we will fill this in the loop below.
        weight_bounds = np.zeros(shape=(n_inputs, 2, in_dims + 1, mid_dims))
        weight_bounds[:, 0, :, :] = -np.Infinity
        weight_bounds[:, 1, :, :] = +np.Infinity
        for input_index in tqdm(range(n_inputs), desc="Computing Bounds"):
            label = labels[input_index]
            for other_label in range(out_dims):
                if other_label == label:
                    continue
                is_met = (original_outputs[input_index, label] >
                          original_outputs[input_index, other_label])
                # At this point, for each weight w, we effectively have two
                # functions in terms of the weight delta 'dw':
                # y1 = a1*dw + o1, y2 = a2*dw + o2
                # We're interested first in their intersection:
                # a1*dw + o1 = a2*dw + o2 => dw = (o2 - o1) / (a1 - a2)
                a_delta = (weight_derivatives[input_index, label] -
                           weight_derivatives[input_index, other_label])
                o_delta = (original_outputs[input_index, label] -
                           original_outputs[input_index, other_label])
                # Shape: (in_dims + 1, mid_dims)
                intersections = -o_delta / a_delta

                # First, we deal with the weights that have non-NaN
                # intersections.
                finite_is, finite_js = np.isfinite(intersections).nonzero()
                finite_bounds = intersections[finite_is, finite_js]
                if is_met:
                    is_upper_bound = finite_bounds >= 0.0
                else:
                    is_upper_bound = finite_bounds <= 0.0

                # First, update the upper bounds.
                upper_bounds = finite_bounds[is_upper_bound]
                upper_is = finite_is[is_upper_bound]
                upper_js = finite_js[is_upper_bound]
                bounds_slice = (input_index, 1, upper_is, upper_js)
                weight_bounds[bounds_slice] = np.minimum(
                    weight_bounds[bounds_slice], upper_bounds)
                # Then, update the lower bounds.
                lower_bounds = finite_bounds[~is_upper_bound]
                lower_is = finite_is[~is_upper_bound]
                lower_js = finite_js[~is_upper_bound]
                bounds_slice = (input_index, 0, lower_is, lower_js)
                weight_bounds[bounds_slice] = np.maximum(
                    weight_bounds[bounds_slice], lower_bounds)

                # Finally, if it is unmet and there are non-finite bounds,
                # NaN-ify those weights.
                if not is_met:
                    nan_is, nan_js = (~np.isfinite(intersections)).nonzero()
                    weight_bounds[input_index, :, nan_is, nan_js] = np.nan
        weight_bounds = weight_bounds.transpose((2, 3, 0, 1))
        to_nanify = weight_bounds[:, :, :, 0] > weight_bounds[:, :, :, 1]
        weight_bounds[to_nanify, :] = np.nan
        # Return (in_dims + 1, mid_dims, n_inputs, {min, max})
        return weight_bounds

    @staticmethod
    def interval_MAX_SMT(intervals):
        """Computes the MAX-SMT for a set of intervals.

        @intervals should be a Numpy array of shape:
        (n_intervals, {lower, upper}).

        The return value is a 3-tuple: (lower, upper, n_met) such that any
        value x satisfying lower <= x <= upper will maximize the MAX-SMT
        problem by meeting n_met constraints.
        """
        lower_indices = np.argsort(intervals[:, 0])
        lower_sorted = intervals[lower_indices, 0]

        upper_indices = np.argsort(intervals[:, 1])
        upper_sorted = intervals[upper_indices, 1]

        best_lower, best_upper = 0, 0
        upper_i = 0
        best_met = -1
        n_met = 0
        for lower_i, lower in enumerate(lower_sorted):
            # First, we update upper -- everything in this loop is an interval
            # we were meeting before but not anymore.
            while upper_sorted[upper_i] < lower:
                n_met -= 1
                upper_i += 1
            # We now meet the interval that this lower is from.
            n_met += 1
            if n_met > best_met:
                best_lower, best_upper = lower, upper_sorted[upper_i]
                best_met = n_met
            elif (len(lower_sorted) - lower_i) < (best_met - n_met):
                # Each iteration adds *at most* 1 to n_met. For us to even have
                # a chance of updating best_met, then, we will have to do at
                # least (best_met - n_met) more iterations.
                break
        return best_lower, best_upper, best_met

    def propose_patch(self, weight_bounds, learn_rate=1.0):
        """Proposes a weight-patch for the patch_layer based on @weight_bounds.

        @weight_bounds should be of shape:
        (in_dims, mid_dims, n_inputs, {min,max})
        """
        in_dims, mid_dims, _, _ = weight_bounds.shape

        best_index = (None, None)
        best_constraints = -1
        best_delta = 0.0
        indices = itertools.product(range(in_dims), range(mid_dims))
        for in_dim, mid_dim in tqdm(indices, total=(in_dims * mid_dims),
                                    desc="Computing Patch"):
            bounds = weight_bounds[in_dim, mid_dim, :, :]
            # We focus on the bounds that are non-NaN
            non_nan_bounds = bounds[~np.isnan(bounds[:, 0])]
            if len(non_nan_bounds) < best_constraints:
                continue
            lower, upper, n_met = self.interval_MAX_SMT(non_nan_bounds)

            if n_met <= best_constraints:
                continue
            best_constraints = n_met
            best_index = (in_dim, mid_dim)

            if lower <= 0.0 <= upper:
                best_delta = 0.0
            else:
                # True if the interval suggests to increase the weight.
                is_increase = lower > 0.0
                # If the interval suggests to increase the weight, suggest a
                # delta slightly above lower. Otherwise, suggest one slightly
                # below upper. Either way, we're trying to stay as close to 0
                # as possible.
                ratio = 0.1 if is_increase else 0.9
                best_delta = lower + (ratio * (upper - lower))
                if not np.isfinite(best_delta):
                    eps = 0.1
                    if is_increase: # => upper == np.Infinity
                        assert np.isfinite(lower + eps)
                        best_delta = lower + eps
                    elif upper < 0.0: # => lower == -np.Infinity
                        assert np.isfinite(upper - eps)
                        best_delta = upper - eps
                    else:
                        assert False
        assert np.isfinite(best_delta)
        print("Would be satisfying", best_constraints, "constraints.")
        print("Updating weight", best_index)
        best_delta *= learn_rate
        return best_index, best_delta, best_constraints

    def construct_patched(self, patched_layer):
        """Constructs a MaskingNetwork given the patched layer.
        """
        activation_layers = self.network.layers
        value_layers = activation_layers.copy()
        value_layers[self.layer_index] = patched_layer
        return MaskingNetwork(activation_layers, value_layers)

    def compute(self, steps):
        """Performs the Network patching for @steps steps.

        Returns the patched network as an instance of MaskingNetwork.
        Intermediate networks (i.e., the network after i steps) are stored in
        self.intermediates.
        """
        if len(self.intermediates) > steps:
            return self.intermediates[steps]

        # Linearize the network around this layer.
        _, layer_inputs, lin_post = self.linearize_around(
            self.inputs, self.network, self.layer_index)
        # We allow restarting from a partially-patched state.
        start_step = len(self.intermediates)
        start_net = self.intermediates[start_step - 1]
        # Keep track of the latest version of the weights/biases as we patch
        # it.
        layer = start_net.value_layers[self.layer_index]
        weights = layer.weights.numpy().copy()
        biases = layer.biases.numpy().copy()
        # We routinely divide by zero and utilize NaN values; Numpy's warning
        # are not particularly useful here.
        np.seterr(divide="ignore", invalid="ignore")
        for step in range(start_step, steps + 1):
            print("Step:", step)
            start_time = timer()
            # First, we compute the interval on each weight that makes each
            # constraint met.
            bounds = self.compute_intervals(layer_inputs, self.labels,
                                            layer, lin_post)
            # Then, we use those intervals to compute the optimal single-weight
            # change.
            index, delta, _ = self.propose_patch(bounds)
            # Finally, we make the update and save the new MaskingNetwork.
            if index[0] < weights.shape[0]:
                weights[index] += delta
            else:
                assert index[0] == weights.shape[0]
                biases[index[1]] += delta
            layer = FullyConnectedLayer(weights.copy(), biases.copy())
            self.intermediates.append(self.construct_patched(layer))
            self.times.append(timer() - start_time)
        np.seterr(divide="warn", invalid="warn")
        return self.intermediates[steps]
