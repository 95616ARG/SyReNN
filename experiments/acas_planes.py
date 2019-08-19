"""Methods for analyzing the ACAS Xu network with planes.
"""
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
from pysyrenn import PlanesClassifier
from experiments.acas_lines import ACASLinesExperiment
from experiments.polar_image import PolarImage

class ACASPlanesExperiment(ACASLinesExperiment):
    """Experiment class to reproduce Section 5 of [2].

    NOTE: Region of interest is specified in experiment/acas_lines.py.
    """
    def prepare_classified(self, classified_plane, input_helpers):
        """Converts a classified plane to a form ready for artifact recording.

        Essentially, it separates the pre/post images and resets the preimages.
        This makes it easier to plot in the Analyze phase.
        """
        pre_planes, labels = classified_plane
        reset_pre_planes = list(map(input_helpers["reset"], pre_planes))
        return reset_pre_planes, labels

    def region_of_interest(self):
        """Returns the region of interest as a V-Polytope.

        The region of interest is defined by self.{min, max}_{x, y}.
        """
        max_rho = np.sqrt(max(self.min_x**2, self.max_x**2) +
                          max(self.min_y**2, self.max_y**2))
        return np.array([
            self.build_input(0, -np.pi),
            self.build_input(max_rho, -np.pi),
            self.build_input(max_rho, np.pi),
            self.build_input(0, np.pi),
        ])

    def load_scenario(self, scenario):
        """Loads a particular scenario into self.

        @scenario should be a 3-tuple:
        (intruder_heading, own_velocity, intruder_velocity)
        """
        attacker_heading_deg, own_velocity, attacker_velocity = scenario
        self.intruder_heading = np.radians(attacker_heading_deg)
        self.own_velocity = own_velocity
        self.intruder_velocity = attacker_velocity

    def run_for_network(self, network_name, data_csv):
        """Gets the plane classifications for a single network and stores them.
        """
        network = self.load_network(network_name)
        input_helpers = self.load_input_data("acas")

        # Rows are {head-on, perpendicular, opposite, -perpendicular}.
        # Columns are {slow, fast}.
        # THESE SHOULD MATCH WITH THOSE IN:
        # ../third_party/eran_preconditions/experiment.py.
        scenarios = [(-180, 150, 150), (-180, 500, 500),
                     (-90, 150, 150), (-90, 500, 500),
                     (0, 150, 150), (0, 500, 500),
                     (90, 150, 150), (90, 500, 500)]

        for scenario in scenarios:
            print("Scenario (Heading, Velocities):", scenario)
            self.load_scenario(scenario)
            scenario_str = "%03d_%03d_%03d" % scenario
            # Do the classification.
            plane = self.region_of_interest()
            processed_plane = input_helpers["process"](plane)

            start_time = timer()
            classifier = PlanesClassifier(network, [processed_plane],
                                          preimages=True)
            classifier.partial_compute()
            fhat_size = len(classifier.transformed_planes[0])
            classified = classifier.compute()[0]
            duration = (timer() - start_time)

            self.record_artifact(
                self.prepare_classified(classified, input_helpers),
                "%s/%s/classified" % (network_name, scenario_str), "pickle")

            self.write_csv(data_csv, {
                "network": network_name,
                "scenario": scenario_str,
                "time": duration,
                "fhat_size": fhat_size,
            })

    def run(self):
        """Run the transformer, split the plane, and save to disk.
        """
        data_csv = self.begin_csv("data",
                                  ["network", "scenario", "time", "fhat_size"])
        self.record_artifact("data", "data", "csv")
        n_models = input("Number of models (# or * for all): ")
        n_models = -1 if n_models == "*" else int(n_models)
        models_run_for = 0
        for i in range(1, 6):
            for j in range(1, 10):
                network_name = "acas_%d_%d" % (i, j)
                print("Running Experiment for Network:", network_name)
                # Now we run the experiment for this layer.
                self.run_for_network(network_name, data_csv)
                models_run_for += 1
                if models_run_for == n_models:
                    return

    def plot_as_cartesian(self, classified, color_fn=None, alpha=1.0):
        """Plots 2D polygons, interpreting vertices as Cartesian coordinates.

        @classified should be a tuple (planes, corresponding_labels) where each
        plane in planes is a Numpy array of vertices defining a polygon.

        @color_fn should be a function taking labels to hex colors. None uses
        self.color.
        """
        color = color_fn if color_fn is not None else self.color
        for pre_plane, label in zip(*classified):
            poly = patches.Polygon(pre_plane[:, :2], facecolor=color(label),
                                   alpha=alpha)
            plt.gca().add_patch(poly)

    def plot(self, intruder_heading, polygons, colors, window_splits,
             artifact_key):
        """Plots the polar-space classified @polygons in Cartesian space.

        Primarily uses experiments/polar_image.py.
        """
        max_rho = np.sqrt(max(self.min_x**2, self.max_x**2) +
                          max(self.min_y**2, self.max_y**2))
        plot = PolarImage((2 * max_rho, 2 * max_rho), (1001, 1001))
        plot.plot_polygons(polygons, colors, window_splits)
        if window_splits > 0:
            # Clean up the ``coloring outside the lines'' from the window plot.
            plot.circle_frame(max_rho, "#ffffff")
        plane = Image.open("plane.png").rotate(90)
        width = 100
        height = int(width * (plane.height / plane.width))
        # NOTE: Pillow uses (width, height), not (height, width).
        plane = plane.resize((width, height))
        plane_array = np.asarray(plane).copy()
        assert plane_array.shape == (height, width, 4)
        # Place the plane in the center of the image.
        plot.place_rgba(plane_array, png_center=(500, 500))
        # Place the intruder plane at the edge of the image.
        plane = plane.rotate(intruder_heading)
        red_plane = self.color_plane_png(np.array(plane), [255, 0, 0], False)
        plot.place_rgba(red_plane, png_center=(200, 750))
        self.record_artifact(plot.image, artifact_key, "rgb_image")

    def analyze(self):
        """Writes plots from the transformed planes.
        """
        max_rho = np.sqrt(max(self.min_x**2, self.max_x**2) +
                          max(self.min_y**2, self.max_y**2))
        print("Plotting results.")
        self.grayscale = input("[G]rayscale or [C]olor? ").lower()[0] == "g"
        also_cartesian = input("Plot Cartesian too? [y/n]: ").lower()[0] == "y"
        data_csv = self.read_artifact("data")
        for row in data_csv:
            path = "{network}/{scenario}".format(**row)
            print("Network:", row["network"])
            print("Scenario (Heading_Velocities):", row["scenario"])
            intruder_heading = int(row["scenario"].split("_")[0])
            classified = self.read_artifact("%s/classified" % (path))
            pre_planes, labels = classified
            pre_planes = [pre_plane[:, :2] for pre_plane in pre_planes]
            classified = (pre_planes, labels)
            color_labels = list(map(self.color, labels))
            self.plot(intruder_heading, pre_planes, color_labels, 20,
                      "%s/polar" % path)

            # Then, take only the polygons with SR.
            single_class = [i for i, label in enumerate(labels) if label == 4]
            single_class_planes = [pre_planes[i] for i in single_class]
            single_class_colors = [color_labels[i] for i in single_class]
            # We *CANNOT* use window-splitting because we don't fill the entire
            # circle.
            self.plot(intruder_heading, single_class_planes,
                      single_class_colors, 0, "%s/polar_single" % path)

            if also_cartesian:
                self.plot_as_cartesian(classified)
                plt.xlim((0, max_rho))
                plt.ylim((-np.pi, np.pi))
                self.record_artifact(plt, "%s/cartesian" % path,
                                     "matplotlib")
                plt.clf()
        return True

if __name__ == "__main__":
    ACASPlanesExperiment("acas_planes").main()
