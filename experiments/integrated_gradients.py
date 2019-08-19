"""Analysis for the integrated gradients lines.
"""
import numpy as np
import svgwrite
from tqdm import tqdm
from pysyrenn import IntegratedGradients
from experiments.experiment import Experiment

class IntegratedGradientsExperiment(Experiment):
    """Runs experiments from Section 4 of [1].

    In that section, we report:
    0. A figure going from black to a 1 with line ticks in place. (Figure 3)
    1. Mean relative error of the m-tilde approach on CIFAR10
       conv{small,medium,big}. (Table 1)
    2. Average number of linear partitions for CIFAR10 conv{small,medium,big}.
       (Table 2)
    2. Average number of samples needed to get within 5% for the next 5 steps
       on CIFAR10 conv{small,medium,big}. Outliers needing >1k samples removed.
       Separate results for {left,right,trap} sampling. (Table 2)
    """
    @staticmethod
    def mean_error(attributions, reference):
        """Computes average relative error between attributions and reference.

        @reference is taken to be the ground truth, while @attributions is
        taken to be the "measurement."
        """
        return np.mean(np.abs(attributions - reference)
                       / np.abs((reference + 10e-12)))

    def n_samples_to_5pct(self, network, baseline, image, label, exact_IG):
        """Returns number of samples needed to be "safely" within 5%.

        Returns a dictionary of the form:

        {"left": n_left, "right": n_right, "trap": n_trap"}

        Where n_{type} is the minimum integer such that sampling with n_{type},
        n_{type} + 1, ..., n_{type} + 4 samples using type {type} all produce
        attribution maps within 5% mean relative error of exact_IG.
        """
        delta = image - baseline

        def grads_for_partitions(partitions):
            """Computes gradients at endpoints for a particular partitioning.
            """
            endpoint_distances = [(i / partitions)
                                  for i in range(partitions + 1)]
            endpoints = baseline + np.outer(endpoint_distances, delta)
            grads = network.compute_gradients(endpoints, label)
            return grads

        def compute_IG_error_from_grads(grads, sample_type):
            """Computes integrated gradients from a set of endpoint gradients.
            """
            if sample_type == "left":
                grads = grads[:-1]
            elif sample_type == "right":
                grads = grads[1:]
            elif sample_type == "trap":
                grads = ((grads[:-1] + grads[1:]) / 2.0)
            else:
                raise NotImplementedError
            IG = delta * np.average(grads, axis=0)
            return self.mean_error(IG, exact_IG)

        def check_for_stable(status_array, around_index):
            """Checks a status-array (see below) for a valid n_{type} index.

            Assumes that any such index will be in the region around
            @around_index +- 6. Returns the index if found, otherwise False.
            """
            in_a_row = 0
            for i in range(around_index - 6, around_index + 6):
                if i > (len(status_array) - 1):
                    break
                if status_array[i] == +1:
                    in_a_row += 1
                else:
                    in_a_row = 0
                if in_a_row == 5:
                    return (i + 1) - in_a_row
            return False

        status = {
            "left": np.zeros(1006),
            "right": np.zeros(1006),
            "trap": np.zeros(10006)
        }
        sample_types = ["left", "right", "trap"]
        n_samples = {"left": None, "right": None, "trap": None}
        running_partitions = 1
        furthest_checked = running_partitions
        back_checking = False
        while running_partitions < 1006:
            furthest_checked = max(furthest_checked, running_partitions)

            # First, check to see if we have found any stability.
            for sample_type in sample_types:
                stable_start = check_for_stable(status[sample_type],
                                                running_partitions)
                if stable_start is not False:
                    n_samples[sample_type] = stable_start
                    if sample_type == "trap":
                        n_samples[sample_type] += 1
                    sample_types.remove(sample_type)

            # Check to see if we're now done.
            if not sample_types:
                break

            # Then, check if we have already checked this partition count.
            if status["left"][running_partitions] != 0:
                running_partitions += 1
                continue

            # Otherwise, run the analysis.
            any_hits = False
            grads = grads_for_partitions(running_partitions)
            for sample_type in sample_types:
                error = compute_IG_error_from_grads(grads, sample_type)
                if error < 0.05:
                    status[sample_type][running_partitions] = +1
                    any_hits = True
                else:
                    status[sample_type][running_partitions] = -1

            if any_hits and not back_checking:
                back_checking = True
                running_partitions -= 5
                running_partitions = max(running_partitions, 1)
            elif any_hits and back_checking:
                running_partitions += 1
            else: # not any_hits
                running_partitions = furthest_checked + 5
                back_checking = False
        return n_samples

    @staticmethod
    def m_tilde_IG(network, baseline, image, label):
        """Computes IG using heuristics from prior work.

        We try increasing the number of samples between 20 and 1001, until the
        heuristics error is within 1%. (Note that the actual prior work
        recommendation is 5%; we found this was too easy to meet and resulted
        in extremely poor accuracy, so we have used a stronger heuristic here.)
        """
        post_baseline, post_image = network.compute([baseline, image])
        exact_sum = (post_image - post_baseline)[label]
        delta = image - baseline

        for steps in range(20, 1001):
            sample_points = [baseline + ((float(i) / steps) * delta)
                             for i in range(0, steps + 1)]
            gradients = network.compute_gradients(sample_points, label)

            avg_grads = np.average(gradients[:-1], axis=0)
            attributions = (image - baseline) * avg_grads
            attribution_sum = np.sum(attributions)

            sum_error = abs((attribution_sum - exact_sum) / exact_sum)
            if sum_error < 0.01:
                return attributions
        return None

    def run_for_image(self, network_name, network, input_data,
                      image_index, result_file):
        """Performs the experiment for a single network/image pair.

        Writes results directly to result_file, which we assume is compatible
        with Experiment.write_row and has the columns [network, image, exact
        regions, m_tilde error, sample left regions, sample right, sample
        tilde].
        """
        image = input_data["raw_inputs"][image_index]
        label = input_data["labels"][image_index]
        result = {"network": network_name, "image": image_index}

        baseline = input_data["process"](np.zeros_like(image))
        image = input_data["process"](image)

        # First, we get the exact IG using SyReNN.
        # In addition to the raw attributions, we need the number of samples
        # taken. To do this, we use the partial-result API to query the raw
        # transformer result before it is used to compute IG.
        IG = IntegratedGradients(network, [(baseline, image)])
        IG.partial_compute()
        result["exact_regions"] = IG.n_samples[0]
        exact_IG = IG.compute_attributions(label)[0]

        # Next, we sample with the "M-Tilde" approach (prior standard).
        m_tilde_IG = self.m_tilde_IG(network, baseline, image, label)
        if m_tilde_IG is None:
            result["m_tilde_error"] = None
        else:
            result["m_tilde_error"] = self.mean_error(m_tilde_IG, exact_IG)

        # Then, {left,right,trap}-samples
        n_samples = self.n_samples_to_5pct(network, baseline, image,
                                           label, exact_IG)
        result["left_samples"] = n_samples["left"]
        result["right_samples"] = n_samples["right"]
        result["trap_samples"] = n_samples["trap"]
        self.write_csv(result_file, result)

    def run(self):
        """Runs the integrated gradients experiment.
        """
        # Then, for each {network,image} run the analysis with exact, m-tilde,
        # and sample-until-5-{left,right,trap}.
        networks = [
            "cifar10_relu_convsmall",
            "cifar10_relu_convmedium",
            "cifar10_relu_convbig_diffai"
        ][:int(input("Number of networks (1 - 3): "))]
        result_file = self.begin_csv(
            "ig_run_data",
            ["network", "image", "exact_regions", "m_tilde_error",
             "left_samples", "right_samples", "trap_samples"])
        self.record_artifact("ig_run_data", "ig_run_data", "csv")

        for network_name in networks:
            print("Running experiments on network:", network_name)
            network = self.load_network(network_name)
            is_eran_conv_model = "conv" in network_name
            input_data = self.load_input_data("cifar10_test", is_eran_conv_model)
            n_inputs = len(input_data["raw_inputs"])
            for i in tqdm(range(n_inputs)):
                self.run_for_image(network_name, network, input_data, i,
                                   result_file)

    def figure_3(self):
        """Produces Figure 3 from [1].
        """
        network = self.load_network("mnist_relu_3_100")
        data = self.load_input_data("mnist_test", is_eran_conv_model=False)

        image_index = 0
        image = data["raw_inputs"][image_index]
        baseline = np.zeros_like(image)

        endpoint_ratios = network.exactline(
            baseline, image, compute_preimages=False, include_post=False)

        color = "gray"

        fig = svgwrite.Drawing(profile="full")
        linewidth = 100
        pad = 3
        fig.add(fig.image(
            self.image_to_datauri(self.rgbify_image(baseline)),
            (0, 0), size=(32, 32)))
        fig.add(fig.image(
            self.image_to_datauri(self.rgbify_image(image)),
            (32 + linewidth + (2 * pad), 0), size=(32, 32)))

        fig.add(fig.line((32 + pad, 16), (32 + pad + linewidth, 16),
                         stroke=color, stroke_width=0.5))
        for ratio in endpoint_ratios[1:-1]:
            x = 32 + pad + (ratio * linewidth)
            fig.add(fig.line((x, 5), (x, 27), stroke=color, stroke_width=0.5))
        self.record_artifact(fig, "expository_figure", "svg")

    def analyze(self):
        """Produces Figure 3 and summaries for tables 1 and 2.
        """
        # Produce Figure 3 first.
        self.figure_3()

        def print_summary(rows, key, name):
            data = np.array([float(row[key]) for row in rows])
            print("%s: %s" % (name, self.summarize(data)))

        result_rows = self.read_artifact("ig_run_data")
        networks = set(row["network"] for row in result_rows)
        for network in networks:
            print("~~~~~~ Network: %s ~~~~~~" % network)
            net_rows = [row for row in result_rows
                        if row["network"] == network]
            print("Number of inputs:", len(net_rows))
            net_rows = [row for row in net_rows
                        if all(value and value != "None"
                               for value in row.values())]
            print("Number of non-timed-out inputs:", len(net_rows))

            # Data for a row in Table 1.
            print_summary(net_rows, "m_tilde_error", "M-Tilde Error")

            # Data for a row in Table 2.
            print_summary(net_rows, "exact_regions", "Exact Regions")
            print_summary(net_rows, "left_samples", "Left Samples")
            print_summary(net_rows, "right_samples", "Right Samples")
            print_summary(net_rows, "trap_samples", "Trap Samples")
        # We added Figure 3, so we need to re-tar the directory.
        return True

if __name__ == "__main__":
    IntegratedGradientsExperiment("integrated_gradients").main()
