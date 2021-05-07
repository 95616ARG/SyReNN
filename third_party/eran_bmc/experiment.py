"""Demonstrates imprecision when using ERAN for BMC.

Code adapted from the ERAN project: https://github.com/eth-sri/eran
"""
import sys
sys.path.insert(0, "../ELINA/python_interface/")
sys.path.insert(0, ".")
from PIL import Image
import math
import numpy as np
import matplotlib
import matplotlib.image
import os
from eran import ERAN
from fppoly import *
from elina_coeff import *
from elina_linexpr0 import *
from read_net_file import *
from analyzer import *
import tensorflow as tf
import csv
import time
import argparse
from timeit import default_timer as timer
from tqdm import tqdm

args = {
    "complete": False,
    "timeout_lp": 1,
    "timeout_milp": 1,
    "use_area_heuristic": True,
}


def main():
    """Runs the ERAN analysis on the pendulum_continuous model.

    This happens in a few steps:
    1. ERAN doesn't specifically support HardTanh layers, so we translate the
       HardTanh into an equivalent set of ReLU layers using convert_htanh().
    2. We then read the controller network into an ERAN model.
    3. We use ERAN/DeepPoly to extract an abstract value describing the
       network's behavior over the initial set. Module float imprecision
       handling, this abstract value is basically two affine transform A and B
       such that Ax <= f(x) <= Bx for all x in the initial set.
    4. We compute Ax and Bx for a particular point (0.35, 0.35) right on the
       edge of the initial set, and show that the range determined by DeepPoly
       (even after applying a concrete HardTanh at the end) is wide enough to
       mark that point as unsafe even on the first iteration.
    """
    # (1) Translate the model into an equivalent one without HardTanh.
    with_htanh_filename = sys.argv[1]
    no_htanh_filename = "/ovol/pendulum_continuous.no_htanh.eran"
    convert_htanh(with_htanh_filename, no_htanh_filename)
    # (2) Read it into ERAN.
    num_pixels = 2
    model, _, _, _ = read_net(no_htanh_filename, num_pixels, False)
    eran = ERAN(model)

    # (3) Extract an abstract value over the initial set.
    # (3a) Load model and init set into ERAN.
    initLB = np.array([-0.35, -0.35])
    initUB = np.array([0.35, 0.35])

    nn = layers()
    nn.specLB = initLB
    nn.specUB = initUB
    execute_list = eran.optimizer.get_deeppoly(initLB, initUB)
    # NOTE: 9 is just a placeholder specnumber to tell it we're using
    # ACAS.
    analyzer = Analyzer(execute_list, nn, "deeppoly", args["timeout_lp"],
                        args["timeout_milp"], 9, args["use_area_heuristic"])

    # (3b) Perform the analysis and extract the abstract values.
    element, _, _ = analyzer.get_abstract0()

    lexpr = get_lexpr_for_output_neuron(analyzer.man, element, 0)
    uexpr = get_uexpr_for_output_neuron(analyzer.man, element, 0)

    lexpr = np.array(extract_from_expr(lexpr))
    uexpr = np.array(extract_from_expr(uexpr))

    # (3c) Extract the output range for initLB based on the abstract value.
    lower_bound, upper_bound = compute_output_range(initLB, lexpr, uexpr)

    # Apply extra knowledge that -1 <= lower_bound <= upper_bound <= 1.
    lower_bound = max(lower_bound, -1.)
    upper_bound = min(upper_bound, 1.)

    post_lower, post_upper = post_bounds(initLB, lower_bound, upper_bound)
    post_lower, post_upper = post_lower.flatten(), post_upper.flatten()

    lower_safe = np.min(post_lower) >= -0.35
    upper_safe = np.max(post_upper) <= 0.35
    is_safe = lower_safe and upper_safe
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if not is_safe:
        print("ERAN reported initLB unsafe after the first step.")
    else:
        print("Something changed; ERAN used to report initLB unsafe, but now"
              "it says it's safe.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    elina_abstract0_free(analyzer.man, element)

def convert_htanh(with_htanh_filename, no_htanh_filename):
    """Converts a network using HardTanh to one using only ReLUs.

    @with_htanh_filename is the path to the ERAN file describing the network
    using HardTanh, while @no_htanh_filename is the path to the ERAN file this
    function should write the ReLU-only version to.
    """
    # y = ReLU(x + 1)
    # z = ReLU(-y + 2)
    with open(with_htanh_filename, "r") as original_network:
        with open(no_htanh_filename, "w") as no_htanh_network:
            in_hard_tanh = False
            weights = None
            biases = None
            for line in original_network:
                if line.strip() == "HardTanh":
                    in_hard_tanh = True
                    no_htanh_network.write("ReLU\n")
                elif in_hard_tanh and not weights:
                    weights = line
                    no_htanh_network.write(line)
                elif in_hard_tanh and not biases:
                    # HTanh(x) = -ReLU(-ReLU(x + 1) + 2) + 1
                    assert "," not in line
                    bias = float(line.strip("\n[]"))
                    no_htanh_network.write("[{}]\n".format(bias + 1.0))
                    no_htanh_network.write("ReLU\n")
                    no_htanh_network.write("[[-1.0]]\n")
                    no_htanh_network.write("[2.0]\n")
                    no_htanh_network.write("Affine\n")
                    no_htanh_network.write("[[-1.0]]\n")
                    no_htanh_network.write("[1.0]\n")
                    in_hard_tanh = False
                else:
                    no_htanh_network.write(line)

def compute_output_range(point, lexpr, uexpr):
    """Computes the range of possible outputs at @point.

    lexpr[:, 0] gives lower bounds on the values of A while lexpr[:, 1] gives
    upper bounds on the values of A. Similarly for uexpr and B.

    Together, they form A, B such that Ax <= f(x) <= Bx.

    This function computes a lower bound on Ax and an upper bound on Bx.
    """
    lower_bound = np.min(lexpr[-1, :])
    for i in range(point.size):
        assert np.sign(lexpr[i, 0]) == np.sign(lexpr[i, 1])
        if np.sign(lexpr[i, 0]) == np.sign(point[i]):
            lower_bound += point[i] * np.min(lexpr[i, :])
        else:
            lower_bound += point[i] * np.max(lexpr[i, :])
    upper_bound = np.max(uexpr[-1, :])
    for i in range(point.size):
        assert np.sign(uexpr[i, 0]) == np.sign(uexpr[i, 1])
        if np.sign(uexpr[i, 0]) == np.sign(point[i]):
            upper_bound += point[i] * np.max(uexpr[i, :])
        else:
            upper_bound += point[i] * np.min(uexpr[i, :])
    return lower_bound, upper_bound

def post_bounds(original_state, action_lower_bound, action_upper_bound):
    """Finds the tightest bounds on the post-state given bounds on the action.

    A, B are environment descriptions for the Pendulum model. See
    ../../experiments/vrl_models.py for more details. Notably, B is positive so
    we get a lower bound by multiplying action_lower_bound and an upper bound
    by multiplying action_upper_bound.
    """
    A = 0.01 * np.array([[0., 1.], [10.0/1.0, 0.]])
    B = 0.01 * 15.0 * np.array([[0.], [1.0]])
    delta_B_lower = action_lower_bound * B
    delta_B_upper = action_upper_bound * B
    original_state = np.array([original_state]).transpose()
    delta_lower = (np.matmul(A, original_state) + delta_B_lower)
    delta_upper = (np.matmul(A, original_state) + delta_B_upper)
    post_lower = original_state + delta_lower
    post_upper = original_state + delta_upper
    return post_lower, post_upper

def extract_from_expr(expr, coeffs=2):
    """Helper method to extract a vector from the ERAN internal representation.

    It returns a vector of size (n + 1, 2), where the last row is the bias and
    vec[:, 0] = inf, vec[:, 1] = sup for the coefficients (used to handle
    floating point imprecision).
    """
    coefficients = []
    for i in range(coeffs):
        coeff = elina_linexpr0_coeffref(expr, i)
        assert coeff.contents.discr == 1
        interval = coeff.contents.val.interval.contents
        assert interval.inf.contents.discr == 0
        assert interval.sup.contents.discr == 0
        inf = interval.inf.contents.val.dbl
        sup = interval.sup.contents.val.dbl
        coefficients.append([inf, sup])
    cst = elina_linexpr0_cstref(expr)
    assert cst.contents.discr == 1
    interval = cst.contents.val.interval.contents
    assert interval.inf.contents.discr == 0
    assert interval.sup.contents.discr == 0
    inf = interval.inf.contents.val.dbl
    sup = interval.sup.contents.val.dbl
    coefficients.append([inf, sup])
    return np.array(coefficients)

if __name__ == "__main__":
    main()
