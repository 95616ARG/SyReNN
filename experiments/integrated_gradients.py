"""Analysis for the integrated gradients lines.
"""
import numpy as np
import svgwrite
from tqdm import tqdm
from pysyrenn import IntegratedGradients
from experiments.experiment import Experiment
import experiments.integral_approximations as integral_approximations
import gc

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

    Note that the cifar10_relu_convbig_diffai model can use large amounts of
    memory when computing all desired gradients at once. To prevent the script
    from running out of memory we have made sure that everywhere we compute a
    potentially-large number of gradients that we batch the computations
    effectively and explicitly free large chunks of memory as soon as possible.
    """
    @staticmethod
    def mean_error(attributions, reference):
        """Computes average relative error between attributions and reference.

        @reference is taken to be the ground truth, while @attributions is
        taken to be the "measurement."
        """
        return np.mean(np.abs(attributions - reference)
                       / np.abs((reference + 10e-12)))

    def batched_IG(self, network, baseline, delta, label,
                   sample_ratios, weights):
        """Efficiently approximates IG with multiple approximation methods.

        NOTE: This function actually returns multiple different approximations,
        one for each list in @weights. The idea is that @weights will be a
        list-of-lists-of-floats, with each sub-list corresponding to a single
        sampling method. So you can compute the approximations for all methods
        simultaneously.

        Arguments
        =========
        - @network is the network to compute the IG for.
        - @baseline is the baseline to use.
        - @delta is (@image - @baseline), pixel-wise.
        - @label is the index of the output to use for computing the gradient.
        - @sample_ratios is a list of floats between 0.0 and 1.0, indicating
          the ratios from baseline -> image that should be used for sampling.
        - @weights is a list-of-list-of-floats. Each inner list should have the
          same length as @sample_ratios, be non-negative, and (for most
          approximation methods) sum to 1.0.

        Return Value
        ============
        A list of Numpy arrays of the same shape as @baseline, one per sub-list
        in @weights. The ith return value corresponds to the approximation
        using weights[i].
        """
        attributions = [np.zeros_like(baseline) for _ in weights]
        for batch_start in range(0, len(sample_ratios), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_ratios = sample_ratios[batch_start:batch_end]

            sample_points = baseline + np.outer(batch_ratios, delta)
            gradients = network.compute_gradients(sample_points, label)
            for i in range(len(weights)):
                batch_weights = weights[i][batch_start:batch_end]
                attributions[i] += np.dot(batch_weights, gradients)
            del gradients
            gc.collect()
        for i in range(len(attributions)):
            attributions[i] *= delta
        return attributions

    def n_samples_to_5pct(self, network, baseline, image, label, exact_IG):
        """Returns number of samples needed to be "safely" within 5%.

        Returns a dictionary of the form:

        dict({method: n_{method} for method in self.sample_types})

        Where n_{method} is the minimum integer such that sampling with
        n_{method}, n_{method} + 1, ..., n_{method} + 4 samples using type
        {method} all produce attribution maps within 5% mean relative error of
        exact_IG.
        """
        delta = image - baseline

        def compute_IG_error(n_partitions, sample_types):
            """Returns the mean errors when using @n_partitions partitions.

            @sample_types should be a list of strings, eg. ["left", "right"].
            The return value is a list with the same length as @sample_types,
            each entry being a float with the relative error of using that
            sampling method with the specified number of partitions.
            """
            ratios, weights = integral_approximations.parameters(n_partitions,
                                                                 sample_types)
            attributions = self.batched_IG(
                network, baseline, delta, label, ratios, weights)
            return [self.mean_error(approx_IG, exact_IG)
                    for approx_IG in attributions]

        def check_for_stable(status_array, around_index):
            """Checks a status-array (see below) for a valid n_{type} index.

            Assumes that any such index will be in the region around
            @around_index +- 6.

            Returns the index if found, otherwise False.
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
            "left": np.zeros(1006, dtype=np.uint8),
            "right": np.zeros(1006, dtype=np.uint8),
            "trap": np.zeros(1006, dtype=np.uint8),
            # If the user selects the option prompted in self.run() then these
            # two entries will be ignored (as they won't be in
            # self.sample_types).
            "simpson": np.zeros(1006, dtype=np.uint8),
            "gauss": np.zeros(1006, dtype=np.uint8),
        }
        sample_types = self.sample_types.copy()
        n_samples = dict({sample_type: None for sample_type in sample_types})
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
                    # These methods use all of the endpoints, while the
                    # "n_samples" we've been keeping track of is really the
                    # number of partitions. So we add one to the get the actual
                    # number of samples used.
                    if sample_type in ("trap", "simpson", "gauss"):
                        n_samples[sample_type] += 1
                    # Stop checking this sampling method on subsequent
                    # iterations.
                    sample_types.remove(sample_type)

            # If we've finished all sampling methods, we're done!
            if not sample_types:
                break

            # If we've already looked at this partition count, we don't need to
            # re-visit it.
            if status[sample_types[0]][running_partitions] != 0:
                running_partitions += 1
                continue

            # Otherwise, run the analysis.
            any_hits = False
            errors = compute_IG_error(running_partitions, sample_types)
            for sample_type, error in zip(sample_types, errors):
                if error < 0.05:
                    status[sample_type][running_partitions] = +1
                    any_hits = True
                else:
                    status[sample_type][running_partitions] = -1

            if any_hits and not back_checking:
                # This is the first we've seen in a row to succeed for one of
                # the sample types; go back and try to see if the previous 5
                # are good as well.
                back_checking = True
                running_partitions -= 5
                running_partitions = max(running_partitions, 1)
            elif any_hits and back_checking:
                # We found one previously and this one is also good; keep going
                # on this group of 5.
                running_partitions += 1
            else:
                # This partition count doesn't satisfy for any of the sample
                # types; skip ahead to 5 past the furthest we've checked so
                # far.
                running_partitions = furthest_checked + 5
                back_checking = False
        return n_samples

    def m_tilde_IG(self, network, baseline, image, label):
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
            ratios, weights = integral_approximations.parameters(steps,
                                                                 ["left"])
            attributions = self.batched_IG(network, baseline, delta, label,
                                           ratios, weights)[0]

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
        IG = IntegratedGradients(network, [(baseline, image)],
                                 batch_size=self.batch_size)
        IG.partial_compute()
        result["exact_regions"] = IG.n_samples[0]
        exact_IG = IG.compute_attributions(label)[0]
        del IG
        gc.collect()

        # Next, we sample with the "M-Tilde" approach (prior standard).
        m_tilde_IG = self.m_tilde_IG(network, baseline, image, label)
        if m_tilde_IG is None:
            result["m_tilde_error"] = None
        else:
            result["m_tilde_error"] = self.mean_error(m_tilde_IG, exact_IG)

        # Then, {left,right,trap}-samples
        n_samples = self.n_samples_to_5pct(network, baseline, image,
                                           label, exact_IG)
        result["left_samples"] = n_samples.get("left", "")
        result["right_samples"] = n_samples.get("right", "")
        result["trap_samples"] = n_samples.get("trap", "")
        result["simpson_samples"] = n_samples.get("simpson", "")
        result["gauss_samples"] = n_samples.get("gauss", "")
        self.write_csv(result_file, result)

    def run(self):
        """Runs the integrated gradients experiment.
        """
        # Then, for each {network,image} run the analysis with exact, m-tilde,
        # and sample-until-5-{left,right,trap[,simp,gauss]}.
        networks = [
            "cifar10_relu_convsmall",
            "cifar10_relu_convmedium",
            "cifar10_relu_convbig_diffai"
        ][:int(input("Number of networks (1 - 3): "))]

        sample_types = ["left", "right", "trap", "simpson", "gauss"]
        fancy_approx = input("Include Simpson and Gaussian Quadrature (y/n)? ")
        if fancy_approx.lower()[0] == "y":
            self.sample_types = sample_types
        else:
            self.sample_types = sample_types[:3]

        batch_sizes = input("Batch sizes (one per network or * for default): ")
        if batch_sizes == "*":
            batch_sizes = [4096, 4096, 256]
        else:
            batch_sizes = list(map(int, batch_sizes.split(",")))

        result_file = self.begin_csv(
            "ig_run_data",
            ["network", "image", "exact_regions", "m_tilde_error",
             "left_samples", "right_samples", "trap_samples",
             "simpson_samples", "gauss_samples"])
        self.record_artifact("ig_run_data", "ig_run_data", "csv")

        for network_name, batch_size in zip(networks, batch_sizes):
            print("Running experiments on network:", network_name)
            self.batch_size = batch_size
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
            """Prints a summary of the rows for a particular column (key).
            """
            data = np.array([float(row[key]) for row in rows if row[key]])
            if not data.size:
                return
            print("%s: %s" % (name, self.summarize(data)))

        def good_row(row):
            """True iff the row has not timed out on the 'core' experiments.
            """
            require_keys = [
                "m_tilde_error",
                "exact_regions",
                "left_samples",
                "right_samples",
                "trap_samples",
            ]
            values = [row[key] for key in require_keys]
            return all(value and value != "None" for value in values)

        result_rows = self.read_artifact("ig_run_data")
        networks = set(row["network"] for row in result_rows)
        for network in networks:
            print("~~~~~~ Network: %s ~~~~~~" % network)
            net_rows = [row for row in result_rows
                        if row["network"] == network]
            print("Number of inputs:", len(net_rows))
            net_rows = [row for row in net_rows if good_row(row)]
            print("Number of non-timed-out inputs:", len(net_rows))
            if not net_rows:
                continue

            # Data for a row in Table 1.
            print_summary(net_rows, "m_tilde_error", "M-Tilde Error")

            # Data for a row in Table 2.
            print_summary(net_rows, "exact_regions", "Exact Regions")
            print_summary(net_rows, "left_samples", "Left Samples")
            print_summary(net_rows, "right_samples", "Right Samples")
            print_summary(net_rows, "trap_samples", "Trap Samples")
            print_summary(net_rows, "simpson_samples", "Simp. Samples")
            print_summary(net_rows, "gauss_samples", "Gauss Samples")
        # We added Figure 3, so we need to re-tar the directory.
        return True

if __name__ == "__main__":
    IntegratedGradientsExperiment("integrated_gradients").main()
