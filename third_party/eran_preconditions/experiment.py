"""Constructs precondition/decision-boundary plots with ERAN.

Code adapted from the ERAN project:
https://github.com/eth-sri/eran
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
from polar_image import PolarImage
from analyzer import *
import tensorflow as tf
import csv
import time
import argparse
from timeit import default_timer as timer
from tqdm import tqdm

def process(x):
    mins = np.array([0.0, -3.141593, -3.141593, 100.0, 0.0])
    maxes = np.array([60760.0, 3.141593, 3.141593, 1200.0, 1200.0])
    means = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    std_deviations = np.array([60261.0, 6.28318530718, 6.28318530718,
                               1100.0, 1200.0])
    return (np.clip(x, mins, maxes) - means) / std_deviations

def reset(y):
    mins = np.array([0.0, -3.141593, -3.141593, 100.0, 0.0])
    maxes = np.array([60760.0, 3.141593, 3.141593, 1200.0, 1200.0])
    means = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    std_deviations = np.array([60261.0, 6.28318530718, 6.28318530718,
                               1100.0, 1200.0])
    return (y * std_deviations) + means

args = {
    "complete": False,
    "timeout_lp": 1,
    "timeout_milp": 1,
    "use_area_heuristic": True,
}

netname = sys.argv[1]
filename, file_extension = os.path.splitext(netname)

num_pixels = 5
model, is_conv, means, stds = read_net(netname, num_pixels, False)
eran = ERAN(model)

def extract_from_expr(expr, coeffs=5):
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

def polytope_for(lexprs, uexprs, label):
    # We have [out]lexpr <= [out_index] <= [out]uexpr
    # and [j]lexpr <= [j] <= [j]uexpr
    # We want to guarantee that [out_index] <= [j], so we need to have:
    # [out]uexpr <= [j]lexpr
    # <->
    # [out]uexpr - [j]lexpr <= 0
    constraints = []
    for j in range(5):
        if j == label:
            continue
        constraints.append(uexprs[label] - lexprs[j])
    constraints = np.array(constraints)
    A_ub = constraints[:, :-1]
    b_ub = -constraints[:, -1]
    return A_ub, b_ub

def color(label):
    """Returns a hex color corresponding to the integer label @label.
    
    From ../../experiments/acas_lines.py
    """
    return ("#4e73b0", "#fdb863", "#b2abd2", "#e66101", "#5e3c99")[label]

def add_planes(image, intruder_heading):
    assert image.image.shape == (1001, 1001, 3)
    plane = Image.open("/eran/plane.png").rotate(90)
    width = 100
    height = int(width * (plane.height / plane.width))
    # NOTE: Pillow uses (width, height), not (height, width).
    plane = plane.resize((width, height))
    plane_array = np.asarray(plane)
    assert plane_array.shape == (height, width, 4)
    # Place the plane in the center of the image.
    image.place_rgba(plane_array, png_center=(500, 500))
    # Place the intruder plane at the edge of the image.
    plane_array = np.array(plane.rotate(intruder_heading))
    cutoff = [200, 200, 200]
    black_indices = np.all((plane_array[:, :, :3] <= cutoff), axis=2)
    plane_array[black_indices, :3] = [255, 0, 0]
    image.place_rgba(plane_array, png_center=(200, 750))

def plot_polar(specLB, specUB, polytopes, image):
    origin_y, origin_x = image.plot_origin
    # Make a box around the region.
    container = [specLB.copy(), specLB.copy(), specUB.copy(), specUB.copy()]
    container[1][0] = specUB[0]
    container[3][0] = specLB[0]
    intruder_heading = np.degrees(reset(specLB)[2])
    box_y, box_x = image.polar_cartesian_box(reset(container)[:, :2])
    png_y_start, png_x_start = image.plot_to_png(box_y[0], box_x[0])
    png_y_end, png_x_end = image.plot_to_png(box_y[1], box_x[1])
    png_y_start, png_y_end = sorted((png_y_start, png_y_end))
    png_y_start = int(png_y_start)
    png_x_start = int(png_x_start)
    png_y_end = int(min(math.ceil(png_y_end), image.image.shape[0]))
    png_x_end = int(min(math.ceil(png_x_end), image.image.shape[1]))
    for label, polytope in polytopes:
        label_color = image.hex_to_int(color(label))
        A_ub, b_ub = polytope
        on_vertices = [np.all(np.matmul(A_ub, vertex) <= b_ub)
                       for vertex in container]
        if not any(on_vertices):
            continue
        if all(on_vertices):
            image.plot_polygons([np.array(reset(container))[:, :2]], [color(label)])
            add_planes(image, intruder_heading)
            return
        for png_y in range(png_y_start, png_y_end):
            for png_x in range(png_x_start, png_x_end):
                plot_y, plot_x = image.png_to_plot(png_y, png_x)
                rho = np.linalg.norm((plot_x, plot_y))
                theta = np.arctan2(plot_y, plot_x)
                rho, theta = process([rho, theta, 0.0, 0.0, 0.0])[:2]
                if not (specLB[0] <= rho <= specUB[0]):
                    continue
                if not (specLB[1] <= theta <= specUB[1]):
                    continue
                point = [rho, theta] + list(specLB[2:])
                if np.all(np.matmul(A_ub, point) <= b_ub):
                    image.image[png_y, png_x, :] = label_color
    add_planes(image, intruder_heading)

output_file = open("%s/results.csv" % sys.argv[2], "w")
output_file.write(",".join(["Attacker Heading", "Own Velocity",
                            "Attacker Velocity", "Splits Amount",
                            "Time", "Polar Plot"]) + "\n")
# Rows are {head-on, perpendicular, opposite, -perpendicular}.
# Columns are {slow, fast}.
scenarios = [(-180, 150, 150), (-180, 500, 500),
             (-90, 150, 150), (-90, 500, 500),
             (0, 150, 150), (0, 500, 500),
             (90, 150, 150), (90, 500, 500)]
for scenario in scenarios:
    # NOTE: Set do_plot = True here if you want plots for all scenarios.
    do_plot = (scenario == scenarios[0])
    print("Scenario (Heading, Velocities):", scenario)
    radius = np.sqrt(10000**2 + 6000**2)
    global_specLB = np.array([0.0, np.radians(-180.0), np.radians(scenario[0]),
                              scenario[1], scenario[2]])
    global_specUB = np.array([radius, np.radians(+180.0),
                              np.radians(scenario[0]),
                              scenario[1], scenario[2]])
    global_specLB = process(global_specLB)
    global_specUB = process(global_specUB)
    global_delta = global_specUB - global_specLB
    for split_amt in [25, 55, 100]:
        print("Split amount:", split_amt)
        filename = ("%s/eran_plot_%d_%d_%d_split_%d" %
                    ((sys.argv[2],) + tuple(scenario) + (split_amt,)))
        splits = [split_amt, split_amt]

        # See ../../experiments/{acas_lines, acas_planes}.py
        pixel_size = 1001
        full_image = PolarImage((2*radius, 2*radius),
                                (pixel_size, pixel_size), silent=True)
        single_class_image = PolarImage((2*radius, 2*radius),
                                        (pixel_size, pixel_size), silent=True)

        time_taken = 0.0
        partitions = itertools.product(range(splits[0]), range(splits[1]))
        n_partitions = splits[0] * splits[1]
        for rho_split, theta_split in tqdm(partitions, total=n_partitions):
            specLB = global_specLB.copy()
            specLB[0] = global_specLB[0] + ((rho_split / splits[0]) * global_delta[0])
            specLB[1] = global_specLB[1] + ((theta_split / splits[1]) * global_delta[1])
            specUB = global_specUB.copy()
            specUB[0] = global_specLB[0] + (((rho_split + 1) / splits[0]) * global_delta[0])
            specUB[1] = global_specLB[1] + (((theta_split + 1) / splits[1]) * global_delta[1])

            nn = layers()
            nn.specLB = specLB
            nn.specUB = specUB
            start_time = timer()
            execute_list = eran.optimizer.get_deeppoly(specLB, specUB)
            # NOTE: 9 is just a placeholder specnumber to tell it we're using
            # ACAS.
            analyzer = Analyzer(execute_list, nn, "deeppoly", args["timeout_lp"],
                                args["timeout_milp"], 9, args["use_area_heuristic"])

            element, _, _ = analyzer.get_abstract0()

            lexprs = [get_lexpr_for_output_neuron(analyzer.man, element, i)
                      for i in range(5)]
            uexprs = [get_uexpr_for_output_neuron(analyzer.man, element, i)
                      for i in range(5)]

            lexprs = np.array(list(map(extract_from_expr, lexprs)))
            uexprs = np.array(list(map(extract_from_expr, uexprs)))

            # We just take the mean of the intervals, as they're usually quite
            # small.
            lexprs = np.mean(lexprs, axis=2)
            uexprs = np.mean(uexprs, axis=2)

            polytopes = [(label, polytope_for(lexprs, uexprs, label)) for label in range(5)]

            time_taken += (timer() - start_time)

            if do_plot:
                plot_polar(specLB, specUB, polytopes, full_image)
                matplotlib.image.imsave("%s.png" % filename, full_image.image)
                plot_polar(specLB, specUB,
                           [(l, p) for l, p in polytopes if l == 4],
                           single_class_image)
                matplotlib.image.imsave("%s_single.png" % filename,
                                        single_class_image.image)

            elina_abstract0_free(analyzer.man, element)

        output_file.write("{ahead},{ovel},{avel},{splits},{time},{name}.png\n".format(
            ahead=scenario[0],
            ovel=scenario[1],
            avel=scenario[2],
            splits=split_amt,
            time=time_taken,
            name=filename))
        output_file.flush()
output_file.close()
