"""Helper methods for supporting different integral approximations.

Compare to:
https://github.com/pytorch/captum/blob/master/captum/attr/_utils/approximation_methods.py
"""
import numpy as np

def _parameters(n_partitions, sample_type):
    """Returns the sample points and weights for a particular approximation.
    """
    if sample_type == "left":
        ratios = [i / n_partitions for i in range(n_partitions)]
        weights = [1. / n_partitions for _ in range(n_partitions)]
    elif sample_type == "right":
        ratios = [(i + 1) / n_partitions for i in range(n_partitions)]
        weights = [1. / n_partitions for _ in range(n_partitions)]
    elif sample_type == "trap":
        ratios = [i / n_partitions for i in range(n_partitions + 1)]

        total_weight = 2. * n_partitions
        weights = [2. / total_weight for _ in range(n_partitions - 1)]

        weights.insert(0, 1. / total_weight)
        weights.append(1. / total_weight)
    elif sample_type == "simpson":
        ratios = [i / n_partitions for i in range(n_partitions + 1)]
        weights = [4. if i % 2 == 1 else 2.  for i in range(n_partitions + 1)]
        weights[0] = 1.
        weights[-1] = 1.
        if n_partitions % 2 == 1:
            # Implements the 'avg' rule from scipy.integrate.simps.
            weights[-1] /= 2.
            weights[-2] = (4. + 1.) / 2.
            weights[0] /= 2.
            weights[1] = (4. + 1.) / 2.
        weights = [weight / (3. * n_partitions) for weight in weights]
    elif sample_type == "gauss":
        ratios, weights = np.polynomial.legendre.leggauss(n_partitions + 1)
        # Rescale sample points from [-1, 1]
        ratios = (ratios + 1.) / 2.
        # Rescale weights from sum = 2 to sum = 1.
        weights /= 2.
    else:
        raise NotImplementedError
    return list(ratios), list(weights)

def parameters(n_partitions, sample_types):
    """Returns sample points and weights for a set of approximations.

    This function's output is compatible with batched_IG(...) in
    integrated_gradients.py. We return a tuple with two members:

    1. A sorted list of floats between 0.0 and 1.0; the ratios along the line
       to sample.
    2. A list-of-lists-of-float weights, each one corresponding to a different
       sampling type in @sample_types and each one being the same length as the
       ratio outputs.
    """
    ratios = []
    weights = []
    for sample_type in sample_types:
        type_ratios, type_weights = _parameters(n_partitions, sample_type)
        ratios.append(type_ratios)
        weights.append(type_weights)

    flat_ratios = sorted(set({
        ratio for ratio_list in ratios for ratio in ratio_list
    }))
    for i, (ratio_list, weight_list) in enumerate(zip(ratios, weights)):
        new_weights = []
        for ratio in flat_ratios:
            try:
                new_weights.append(
                    weight_list[ratio_list.index(ratio)]
                )
            except ValueError:
                new_weights.append(0.0)
        weights[i] = new_weights
    return flat_ratios, weights
