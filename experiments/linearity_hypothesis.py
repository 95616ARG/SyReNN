"""Analysis for counting regions.
"""
import numpy as np
import svgwrite
from experiments.experiment import Experiment
from pysyrenn import LinesClassifier

class LinearityHypothesisExperiment(Experiment):
    """Empirically investigating Linear Explanation of Adversarial Examples.

    Corresponds to Section 5 of [1]. A number of different results are taken
    from this experiment:

    - Expository plot (Figure 4)
    - FGSM density/random density for {normal,diffai,pgd}x{mnist,cifar10}
      (Tables 3ab).
    - Normal/{DiffAI,PGD}x{mnist,cifar10} densities for {FGSM,random}
      directions (Tables 4ab).
    - Weighted relative gradient change (lines 240 - 250).
    """
    @staticmethod
    def perturb_l_inf(image, signs, l_inf):
        """Maximally perturbs (un-processed) @image within some l_inf bound.
        """
        distances = np.floor(l_inf * image)
        perturbed = image + (distances * signs)
        return perturbed.clip(0, 255)

    def fgsm(self, network, image, to_label, l_inf, process):
        """Returns an FGSM-perturbed version of @image.

        @process should be the pre-processing function for @network, while
        @to_label is the target label. @l_inf is the l_inf norm we respect.
        """
        signs = np.sign(network.compute_gradients(
            [process(image)], to_label))[0]
        return self.perturb_l_inf(image, signs, l_inf)

    def random_perturbation(self, image, l_inf, seed):
        """Returns a randomly-perturbed version of @image.

        @seed should be based on the image itself so that all networks get the
        same random images.
        """
        np.random.seed(seed)
        signs = np.sign(np.random.choice([-1, 1], image.shape))
        return self.perturb_l_inf(image, signs, l_inf)

    @staticmethod
    def gradient_error(network, line, transformed):
        """Returns the mean error in the gradient assumption made by FGSM.

        Effectively, this computes the relative error between the gradient
        computed at @line["start"] and the gradient within each linear region
        on @line. The Linearity Hypothesis works if these are all similar, i.e.
        if this function returns a small value.
        """
        start = line["start"]
        end = line["end"]
        delta = end - start
        label = line["target_fgsm_label"]

        start_gradient = network.compute_gradients([start], label)[0]

        region_lengths = []
        midpoint_preimages = []
        for start_ratio, end_ratio in zip(transformed[:-1], transformed[1:]):
            midpoint_ratio = (start_ratio + end_ratio) / 2.0
            midpoint_preimages.append(
                line["start"] + (midpoint_ratio * delta))
            region_lengths.append(end_ratio - start_ratio)

        along_gradients = network.compute_gradients(midpoint_preimages, label)
        gradient_errors = [np.mean(np.abs((start_gradient - along_gradient) /
                                          (along_gradient + 1e-12)))
                           for along_gradient in along_gradients]
        return np.average(gradient_errors, weights=region_lengths)


    def run_for_network(self, network_name, dataset):
        """Runs experiments for a particular network.

        @dataset should be the result of calling Experiment.load_input_data.
        """
        l_inf = 0.03
        network = self.load_network(network_name)
        process = dataset["process"]
        inputs = dataset["raw_inputs"]
        labels = dataset["labels"]

        # First, we generate line descriptions for each line we want to pass
        # through this network.
        lines = []
        for i, image in enumerate(inputs):
            label = labels[i]

            actual_label = np.argmax(network.compute([process(image)])[0])
            if actual_label != label:
                # We only consider points which are correctly classified by the
                # network.
                continue

            fgsm_label = (label + 1) % 10
            targets = {
                "fgsm": self.fgsm(network, image, fgsm_label, l_inf, process),
            }
            for j in range(4):
                targets["random%d" % j] = self.random_perturbation(
                    image, l_inf, ((20 * i) + j))

            image_dir = "%s/%03d" % (network_name, i)

            # Save the original image.
            self.record_artifact(self.rgbify_image(image),
                                 "%s/start" % image_dir, "rgb_image")

            # Then make lines to & save all of the other target images.
            for target_name, target_image in targets.items():
                target_image_key = "%s/%s" % (image_dir, target_name)
                self.record_artifact(self.rgbify_image(image),
                                     target_image_key, "rgb_image")

                target_label = np.argmax(
                    network.compute([process(target_image)])[0])
                length = np.linalg.norm(process(target_image) - process(image))

                lines.append({
                    "image": i,
                    "artifact_dir": image_dir,
                    "start": process(image),
                    "end": process(target_image),
                    "type": target_name,
                    "start_label": label,
                    "end_label": target_label,
                    "target_fgsm_label": fgsm_label,
                    "length": length,
                    "gradient_error": None,
                })

        transformed_lines = network.exactlines(
            [(line["start"], line["end"]) for line in lines],
            # We want post to get the 'break' points in the analysis.
            compute_preimages=False, include_post=True)

        for line, transformed in zip(lines, transformed_lines):
            # transformed is just a Numpy array of endpoint ratios because we
            # passed compute_preimages=False and include_post=False
            artifact_key = "%s/%s_transformed" % (
                line["artifact_dir"], line["type"])
            self.record_artifact(transformed, artifact_key, "pickle")
            # We use the gradients from the FGSM direction for the inline
            # mention of the "extended experiment."
            if line["type"] == "fgsm":
                line["gradient_error"] = self.gradient_error(network, line,
                                                             transformed[0])

        line_data_key = "%s/line_data" % network_name
        line_data_out = self.begin_csv(
            line_data_key,
            ["image", "artifact_dir", "type", "start_label", "end_label",
             "target_fgsm_label", "gradient_error", "length"],
            extrasaction="ignore")
        for line in lines:
            self.write_csv(line_data_out, line)
        self.record_artifact(line_data_key, line_data_key, "csv")

    def run(self):
        """Runs the linearity-hypothesis experiment.

        In the run phase, we select a number of lines (some shared by all
        models and some model-specific), then transform them and record the
        relevant statistics.
        """
        dataset_types = ["cifar10", "mnist"]
        network_types = ["relu_convsmall",
                         "relu_convsmall_diffai",
                         "relu_convsmall_pgd"]
        n_networks = input("Number of models (# or * for all): ")
        n_networks = -1 if n_networks == "*" else int(n_networks)
        networks_finished = 0
        network_names = []
        for dataset_type in dataset_types:
            dataset_name = "%s_test" % dataset_type
            for network_type in network_types:
                network_name = "%s_%s" % (dataset_type, network_type)
                network_names.append(network_name)
                print("Running Experiment for Network:", network_name)
                dataset = self.load_input_data(dataset_name,
                                               "conv" in network_name)
                self.run_for_network(network_name, dataset)
                networks_finished += 1
                if networks_finished == n_networks:
                    break
            if networks_finished == n_networks:
                break
        self.record_artifact(network_names, "networks", "pickle")

    @staticmethod
    def group_by_image(lines):
        """Groups lines by their starting ("natural") image.
        """
        groups = {}
        for line in lines:
            try:
                groups[int(line["image"])].append(line)
            except KeyError:
                groups[int(line["image"])] = [line]
        return groups

    def compute_density(self, line):
        """Computes the density for a line.
        """
        endpoints, postimages = self.read_artifact(
            "{artifact_dir}/{type}_transformed".format(**line))
        return len(endpoints) / float(line["length"])

    def classification_regions(self, line):
        """Computes the classification regions for a line.
        """
        transformed = self.read_artifact(
            "{artifact_dir}/{type}_transformed".format(**line))
        classifier = LinesClassifier.from_exactlines([transformed])
        return classifier.compute()[0]

    @staticmethod
    def corresponding_line(line, other_lines):
        """Finds a line in @other_lines that corresponds to @line.

        This is used for, eg., finding the same random-line in another model's
        results.
        """
        try:
            return next(other_line for other_line in other_lines
                        if (other_line["type"] == line["type"] and
                            other_line["image"] == line["image"]))
        except StopIteration:
            return None

    def analyze(self):
        """Analyzes the result of the experiment.
        """
        networks = self.read_artifact("networks")

        # First, we get the FGSM/Random data for all networks.
        print("Table 3 Data (FGSM/Random):")
        for network in networks:
            print("Network:", network)
            lines = self.read_artifact("%s/line_data" % network)
            groups = self.group_by_image(lines)
            ratios = []
            for image_lines in groups.values():
                # There should only be one FGSM line.
                fgsm_line = next(line for line in image_lines
                                 if line["type"] == "fgsm")
                fgsm_density = self.compute_density(fgsm_line)
                # There may be many random lines.
                random_lines = [line for line in image_lines
                                if line["type"].startswith("random")]
                random_densities = map(self.compute_density, random_lines)
                ratios.extend(fgsm_density / random_density
                              for random_density in random_densities)
            print("\t%s" % self.summarize(ratios))
        print("")

        print("Table 4 Data (Normal/{DiffAI,PGD})")
        normal_networks = [network for network in networks
                           if not ("pgd" in network or "diffai" in network)]
        for normal_network in normal_networks:
            print("Normal Network:", normal_network)
            normal_lines = self.read_artifact("%s/line_data" % normal_network)
            normal_fgsm_lines = [line for line in normal_lines
                                 if line["type"] == "fgsm"]
            normal_random_lines = [line for line in normal_lines
                                   if line["type"].startswith("random")]
            for suffix in ["diffai", "pgd"]:
                other_network = "%s_%s" % (normal_network, suffix)
                if other_network not in networks:
                    continue
                other_lines = self.read_artifact("%s/line_data" % other_network)
                other_fgsm_lines = [line for line in other_lines
                                    if line["type"] == "fgsm"]
                other_random_lines = [line for line in other_lines
                                      if line["type"].startswith("random")]
                ratios = []
                for normal_line in normal_fgsm_lines:
                    other_line = self.corresponding_line(normal_line,
                                                         other_fgsm_lines)
                    if other_line is None:
                        continue
                    ratios.append(self.compute_density(normal_line) /
                                  self.compute_density(other_line))
                print("\tFGSM Normal/%s:" % suffix, self.summarize(ratios))
                ratios = []
                for normal_line in normal_random_lines:
                    other_line = self.corresponding_line(normal_line,
                                                         other_random_lines)
                    if other_line is None:
                        continue
                    ratios.append(self.compute_density(normal_line) /
                                  self.compute_density(other_line))
                print("\tRandom Normal/%s:" % suffix, self.summarize(ratios))

            # Reproduce the gradient-error percentage.
            gradient_errors = [float(line["gradient_error"])
                               for line in normal_fgsm_lines]
            print("\tGradient error:", self.summarize(gradient_errors))

        print("Generating versions of Figure 4...")
        self.figure_4(networks)
        return True

    def figure_4(self, networks):
        """Builds expository figures for all valid lines.
        """
        if not ("cifar10_relu_convsmall" in networks and
                "cifar10_relu_convsmall_diffai" in networks):
            print("CIFAR10 convsmall {normal,diffai} networks not run, " +
                  "can't make Figure 4")
            return
        normal_net = "cifar10_relu_convsmall"
        diffai_net = "cifar10_relu_convsmall_diffai"
        normal_lines = self.read_artifact("%s/line_data" % normal_net)
        diffai_lines = self.read_artifact("%s/line_data" % diffai_net)
        normal_images = set(int(line["image"]) for line in normal_lines)
        diffai_images = set(int(line["image"]) for line in diffai_lines)
        both_images = normal_images & diffai_images
        for image in both_images:
            normal_fgsm = next(line for line in normal_lines
                               if (int(line["image"]) == image and
                                   line["type"] == "fgsm"))
            normal_random = next(line for line in normal_lines
                                 if (int(line["image"]) == image and
                                     line["type"] == "random3"))
            diffai_random = self.corresponding_line(normal_random, diffai_lines)
            fig4a, fig4b = self.figure_4_for_lines(normal_fgsm, normal_random,
                                                   diffai_random)
            self.record_artifact(fig4a, "figure_4a_%03d" % image, "svg")
            self.record_artifact(fig4b, "figure_4b_%03d" % image, "svg")

    def figure_4_for_lines(self, normal_fgsm, normal_random, diffai_random):
        """Builds the expository figure for a particular image.
        """
        fig4a = svgwrite.Drawing(profile="full")
        fig4b = svgwrite.Drawing(profile="full")

        def add_line(fig, line, left, top, width, height, color,
                     break_color="blue"):
            endpoints, _ = self.read_artifact(
                "{artifact_dir}/{type}_transformed".format(**line))
            mid = top + (height / 2)
            fig.add(fig.line((left, mid), (left + width, mid),
                             stroke=color, stroke_width=0.5))
            for endpoint in endpoints[1:-1]:
                x = left + (endpoint * width)
                fig.add(fig.line((x, top), (x, top + height),
                                 stroke=color, stroke_width=0.5))
            breaks = self.classification_regions(line)[0][1:]
            for break_distance in breaks:
                x = left + (break_distance[0] * width)
                fig.add(fig.line((x, top - 8), (x, top + height + 8),
                                 stroke=break_color, stroke_width=1.0))

        line_width = 125
        vpad = 3

        def add_start_image(fig, line, y):
            start_key = "{artifact_dir}/start".format(**line)
            start_image = self.read_artifact(start_key)
            start_datauri = self.image_to_datauri(start_image)
            fig.add(fig.image(start_datauri, (0, y), size=(32, 32)))

        def add_end_image(fig, line, y, labels, label_colors):
            end_key = "{artifact_dir}/{type}".format(**line)
            end_image = self.read_artifact(end_key)
            end_datauri = self.image_to_datauri(end_image)
            left = 32 + (2 * vpad) + line_width
            fig.add(fig.image(end_datauri, (left, y), size=(32, 32)))
            for i, label in enumerate(labels):
                fig.add(fig.text(label, insert=(left + 32 + 3, y + (i * 6) + 5),
                                 font_size="5", fill=label_colors[i]))

        labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog",
                  "Frog", "Horse", "Ship", "Truck"]
        # 4a top line is Normal-FGSM.
        add_start_image(fig4a, normal_fgsm, 16)
        fig4a.add(fig4a.text(labels[int(normal_fgsm["start_label"])],
                             insert=(12, 16 + 38), font_size="5"))
        add_end_image(fig4a, normal_fgsm, 0, [
            "Training Method: Normal",
            "Direction: FGSM +3%",
            "Density: %0.2f" % self.compute_density(normal_fgsm),
            "Label: %s" % labels[int(normal_fgsm["end_label"])],
        ], ["black", "black", "black", "black"])
        add_line(fig4a, normal_fgsm, 32 + vpad, 9, line_width, 14, "black")
        # 4a bottom line is Normal-Random
        add_end_image(fig4a, normal_random, 32 + vpad, [
            "Training Method: Normal",
            "Direction: Random +3%",
            "Density: %0.2f" % self.compute_density(normal_random),
            "Label: %s" % labels[int(normal_random["end_label"])],
        ], ["black", "black", "black", "black"])
        add_line(fig4a, normal_random, 32 + vpad, 35 + 9, line_width, 14,
                 "black")

        # 4b top line is Normal-Random.
        add_start_image(fig4b, normal_fgsm, 16)
        fig4b.add(fig4b.text(labels[int(normal_fgsm["start_label"])],
                             insert=(12, 16 + 38), font_size="5"))
        add_end_image(fig4b, normal_fgsm, 0, [
            "Training Method: Normal",
            "Direction: Random +3%",
            "Density: %0.2f" % self.compute_density(normal_random),
            "Label: %s" % labels[int(normal_random["end_label"])],
        ], ["black", "black", "black", "black"])
        add_line(fig4b, normal_random, 32 + vpad, 9, line_width, 14, "black")
        # 4b bottom line is DiffAI-Random
        add_end_image(fig4b, diffai_random, 32 + vpad, [
            "Training Method: DiffAI",
            "Direction: Random +3%",
            "Density: %0.2f" % self.compute_density(diffai_random),
            "Label: %s" % labels[int(diffai_random["end_label"])],
        ], ["green", "green", "green", "green"])
        add_line(fig4b, diffai_random, 32 + vpad, 35 + 9, line_width, 14,
                 "green")
        return fig4a, fig4b

if __name__ == "__main__":
    LinearityHypothesisExperiment("linearity_hypothesis").main()
