"""Methods for analyzing the ACAS Xu network with ExactLine.
"""
import itertools
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pysyrenn
from experiments.experiment import Experiment

class ACASLinesExperiment(Experiment):
    """Experiment class to reproduce Section 3 of [1].

    In that section, we present plots using ExactLine to visualize the decision
    boundary of an ACAS Xu network. Here we produce a superset of the figures
    included in the paper. For each network, we produce a variety of figures:

    - A "vary-theta" figure, which looks like circles emanating from the
      ownship.
    - A "vary-distance" figure, which looks like lines emanating from the
      ownship.
    - An "overlapping" figure, which overlaps the two.
    - A "sample" figure, which samples finitely-many points.
    - A set of "single-line" figures, which help to show what we're doing on a
      more micro level.
    """
    min_x = -6000
    max_x = 10000
    min_y = -6000
    max_y = 6000
    intruder_heading = np.radians(-180.0)
    own_velocity = 150
    intruder_velocity = 150
    grayscale = False

    def color(self, label):
        """Returns a hex color corresponding to the integer label @label.
        """
        if self.grayscale:
            return ("#ffffff", "#555555", "#888888", "#bbbbbb", "#222222")[label]
        # COC WL WR SL SR
        return ("#4e73b0", "#fdb863", "#b2abd2", "#e66101", "#5e3c99")[label]

    def build_input(self, distance, psi):
        """Returns an (un-processed) input point corresponding to the scenario.
        """
        return np.array([distance, psi, self.intruder_heading,
                         self.own_velocity, self.intruder_velocity])

    def sample_points(self, network, input_helpers, sample_along=None):
        """Returns pre/post for finitely-many points.

        By default, this samples in the [min_x, max_x]x[min_y, max_y] region.
        However, when @sample_along = (low, high), both processed, is provided,
        it concentrates all samples on the line between low and high.
        """
        pre_points = []
        post_points = []
        if not sample_along:
            x_samples = np.linspace(self.min_x, self.max_x, 25)
            y_samples = np.linspace(self.min_y, self.max_y, 25)
            for x, y in itertools.product(x_samples, y_samples):
                rho = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                inputs = np.array([rho, theta, self.intruder_heading,
                                   self.own_velocity, self.intruder_velocity])
                pre_points.append(inputs)
                label = np.argmax(
                    network.compute(input_helpers["process"](inputs))[0])
                post_points.append(label)
        else:
            low, high = sample_along
            n_samples = 15
            points = [low + (float(i) / (n_samples - 1))*(high - low)
                      for i in range(n_samples)]
            post_points = np.argmax(network.compute(points), axis=1)
            pre_points = [input_helpers["reset"](point) for point in points]
        return np.array([pre_points, post_points])

    def theta_breaks(self, rho):
        """Returns thetas of intersections with bounding box for a rho-circle.

        One of MatPlotLib's optimizations for plotting arcs seems to make them
        dissappear when they go outside of the bounding box, so this function
        returns subsets of the rho-circle about (0, 0) which are within the
        bounding box.

        This method *ASSUMES* that at least some portion of the theta-circle is
        inside the bounding box.
        """
        # First, we find all x/y intersection points with the min_x and max_x
        # planes.
        intersection_points = [
            (x, sign * np.sqrt(rho**2 - x**2))
            for x, sign in
            itertools.product([self.min_x, self.max_x], [-1, +1])]
        # Then, we add all x/y intersection points with the min_y and max_y
        # planes.
        intersection_points += [
            (sign * np.sqrt(rho**2 - y**2), y)
            for y, sign in
            itertools.product([self.min_y, self.max_y], [-1, +1])]

        # Take only the points that actually exist in the x/y plane.
        intersection_points = [(x, y) for x, y in intersection_points
                               if np.all(np.isreal((x, y)))]

        def in_rectangle(x, y):
            """Returns true if x and y are in the bounding box.
            """
            return ((self.min_x <= x <= self.max_x) and
                    (self.min_y <= y <= self.max_y))

        # Take only the points inside the bounding box.
        intersection_points = [(x, y) for x, y in intersection_points
                               if in_rectangle(x, y)]
        # Find the thetas corresponding to each point.
        intersection_thetas = sorted(np.arctan2(y, x)
                                     for x, y in intersection_points)

        if not intersection_thetas:
            # Either all points are inside or all points are outside; we assume
            # they're all inside.
            return [(np.radians(-180.0), np.radians(180.0))]

        # Check a test point to see if the first intersection-theta is a switch
        # to outside-the-box or inside-the-box.
        mid_start_theta = (np.radians(-180.0) + intersection_thetas[0]) / 2.0
        if in_rectangle(rho * np.cos(mid_start_theta),
                        rho * np.sin(mid_start_theta)):
            intersection_thetas.insert(0, np.radians(-180))
        # Check a test point to see if the last intersection-theta is a switch
        # to outside-the-box or inside-the-box.
        mid_end_theta = (np.radians(-180.0) + intersection_thetas[0]) / 2.0
        if in_rectangle(rho * np.cos(mid_end_theta),
                        rho * np.sin(mid_end_theta)):
            intersection_thetas.append(np.radians(180))

        return list(zip(intersection_thetas[:-1], intersection_thetas[1:]))

    def vary_theta_lines(self, n_lines):
        """Returns (unprocessed) lines varying theta over the boudning box.

        These lines look like concentric circles when plotted in Cartesian
        coordinates.
        """
        max_rho = np.sqrt(
            max(self.min_x**2, self.max_x**2) +
            max(self.min_y**2, self.max_y**2))

        distances = [(float(i + 1) / n_lines) * max_rho
                     for i in range(n_lines)]
        lines = []
        for distance in distances:
            for start, end in self.theta_breaks(distance):
                low = self.build_input(distance, start)
                high = self.build_input(distance, end)
                lines.append((low, high))
        return lines

    def intersection_distance(self, theta):
        """Returns the rho at which the theta-line hits the bounding box.

        This allows us to limit our distance-lines to only the portions that
        are actually in the bounding box.
        """
        intersections = [
            # rho * cos(theta) = max_x
            # rho = max_x / cos(theta)
            self.max_x / np.cos(theta),
            # rho * cos(theta) = min_x
            self.min_x / np.cos(theta),
            # rho * sin(theta) = max_y
            self.max_y / np.sin(theta),
            # rho * sin(theta) = min_y
            self.min_y / np.sin(theta),
        ]
        return min(intersection for intersection in intersections
                   if intersection >= 0.0)

    def vary_distance_lines(self, n_lines,
                            start_theta=-180.0, end_theta=+180.0):
        """Returns lines that vary rho.

        Each line has a different theta parameter.  These look like "spikes"
        coming out of the center when plotted in Cartesian coordinates.
        """
        delta_theta = end_theta - start_theta
        thetas = [start_theta + ((float(i) / (n_lines - 1)) * delta_theta)
                  for i in range(n_lines)]
        thetas = list(map(np.radians, thetas))

        lines = []
        for theta in thetas:
            low = self.build_input(0.0, theta)
            high = self.build_input(self.intersection_distance(theta), theta)
            lines.append((low, high))
        return lines

    @staticmethod
    def prepare_classified(classified_lines, input_helpers):
        """Converts a transformed line to a form ready for artifact recording.

        Essentially, it separates the pre/post images and resets the preimages.
        This makes it easier to plot in the Analyze phase.
        """
        regions = [input_helpers["reset"](region)
                   for region, label in classified_lines]
        labels = [label for region, label in classified_lines]
        return list(zip(regions, labels))

    def get_single_line_data(self, network, input_helpers):
        """Returns data for the single-line plots.

        Note that the single-line plots are handled effectively as a group in
        both the run and analysis phases, in contrast to the full-region plots
        which are each handled individually. This is mostly due to the
        single-line plots all sharing a large amount of metadata (eg. the
        rho/theta used).
        """
        theta_lines = [(
            input_helpers["process"](
                self.build_input(3100, np.radians(-180.0))),
            input_helpers["process"](
                self.build_input(3100, np.radians(+180.0)))
        )]

        distance_lines = [(
            input_helpers["process"](
                self.build_input(0, np.radians(-68.0))),
            input_helpers["process"](
                self.build_input(2 * self.max_y, np.radians(-68.0)))
        )]

        theta_classified = pysyrenn.LinesClassifier(
            network, theta_lines, preimages=True).compute()
        distance_classified = pysyrenn.LinesClassifier(
            network, distance_lines, preimages=True).compute()

        sample_theta = self.sample_points(network, input_helpers,
                                          sample_along=theta_lines[0])
        sample_distance = self.sample_points(network, input_helpers,
                                             sample_along=distance_lines[0])

        return [self.prepare_classified(theta_classified, input_helpers),
                self.prepare_classified(distance_classified, input_helpers),
                sample_theta, sample_distance]

    def run_for_network(self, network_name, theta_lines, distance_lines):
        """
        @theta_lines and @distance_lines should be *processed*
        """
        network = self.load_network(network_name)
        input_helpers = self.load_input_data("acas")

        def save_classified(classified_lines, label):
            self.record_artifact(
                self.prepare_classified(classified_lines, input_helpers),
                "%s/%s" % (network_name, label), "pickle")

        # Now, do the classification.
        theta_classified = pysyrenn.LinesClassifier(
            network, theta_lines, preimages=True).compute()
        save_classified(theta_classified, "theta")

        distance_classified = pysyrenn.LinesClassifier(
            network, distance_lines, preimages=True).compute()
        save_classified(distance_classified, "distance")

        sample_points = self.sample_points(network, input_helpers)
        self.record_artifact(sample_points, "%s/sample" % network_name,
                             "np_array")

        single_line_data = self.get_single_line_data(network, input_helpers)
        self.record_artifact(single_line_data,
                             "%s/single_lines" % network_name, "pickle")

    def run(self):
        """Run the transformer, split the lines, and save to disk.
        """
        n_lines = 100
        input_helpers = self.load_input_data("acas")

        # Precompute the lines we use.
        theta_lines = self.vary_theta_lines(n_lines // 4)
        theta_lines = input_helpers["process"](theta_lines)
        distance_lines = self.vary_distance_lines(n_lines)
        distance_lines = input_helpers["process"](distance_lines)

        n_models = input("Number of models (# or * for all): ")
        n_models = -1 if n_models == "*" else int(n_models)
        models_run_for = 0
        for i in range(1, 6):
            for j in range(1, 10):
                network_name = "acas_%d_%d" % (i, j)
                print("Running Experiment for Network:", network_name)
                # Now we run the experiment for this layer.
                self.run_for_network(network_name, theta_lines, distance_lines)
                models_run_for += 1
                if models_run_for == n_models:
                    return

    @staticmethod
    def attacker_position(inputs):
        """Returns x/y position from an (unprocessed) input scenario.
        """
        rho, theta, _, _, _ = inputs
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    @staticmethod
    def color_plane_png(plane, color, is_normalized):
        """Replaces black/dark-grey pixels in @plane with @color.

        @plane should be a 3D Numpy array (HW{RGB[A]}).
        @color should be a 3-valued Numpy array (HW{RGB}) to use as the new color.
        Used by this class and acas_planes.py to color the intruder red.
        @is_normalized=True means the values are in the range [0.0, 1.0], False
        means the values are in the range [0, 255].
        """
        plane = plane.copy()
        cutoff = np.array([200, 200, 200])
        if is_normalized:
            cutoff = cutoff.astype(np.float32) / 255.0
        black_indices = np.all((plane[:, :, :3] <= cutoff), axis=2)
        plane[black_indices, :3] = color
        return plane

    def finalize_plot(self, artifact_name, attacker_x=None, attacker_y=None):
        """Finalizes and records the current plot as an artifact.
        """
        # Plot the axis ticks.
        plt.ylim((self.min_y - 10.0, self.max_y + 10.0))
        plt.xlim((self.min_x - 10.0, self.max_x + 10.0))
        plt.xticks([self.min_x + 1000, 0.0, self.max_x], size=15)
        plt.yticks([self.min_y + 1000, 0.0, self.max_y], size=15)
        # Add and place the labels.
        ax = plt.gca()
        plt.ylabel("Crossrange (ft)", size=15)
        plt.xlabel("Downrange (ft)", size=15)
        plt.subplots_adjust(bottom=0.25, left=0.25)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # Place the plane.
        plane = plt.imread("plane.png").transpose((1, 0, 2))
        width = (self.max_x - self.min_x) / 10
        height = (496.0 / 499.0) * width
        x_start = -(width / 2.0)
        y_start = -(height / 2.0)
        plt.imshow(plane, extent=[x_start, x_start + width,
                                  y_start, y_start + height], zorder=100)
        plane = np.flip(plane, 1)
        if attacker_x is None:
            attacker_x = self.max_x - (2 * width)
        if attacker_y is None:
            attacker_y = self.max_y - (2 * height)
        red_plane = self.color_plane_png(plane, [1.0, 0, 0], True)
        plt.imshow(red_plane, zorder=100,
                   extent=[attacker_x, attacker_x + width,
                           attacker_y, attacker_y + height])
        self.record_artifact(plt, artifact_name, "matplotlib")
        plt.clf()

    def theta_plot(self, classified_lines, line_width=3.0):
        """Plots a set of transformed lines that vary theta.

        These look like concentric circles.
        """
        for regions, labels in classified_lines:
            for region, label in zip(regions, labels):
                start, end = region

                rho = start[0]
                assert np.isclose(rho, end[0])

                start_theta = np.degrees(start[1])
                end_theta = np.degrees(end[1])

                arc = patches.Arc((0, 0), 2*rho, 2*rho,
                                  theta1=start_theta, theta2=end_theta,
                                  color=self.color(int(label)),
                                  linewidth=line_width)
                plt.gca().add_patch(arc)

    def distance_plot(self, classified_lines):
        """Plots a set of transformed lines that vary rho.

        These look like "spikes".
        """
        for regions, labels in classified_lines:
            for region, label in zip(regions, labels):
                start = self.attacker_position(region[0])
                end = self.attacker_position(region[1])
                plt.plot([start[0], end[0]], [start[1], end[1]],
                         color=self.color(int(label)))


    def overlapping_plot(self, distance_classified, theta_classified):
        """Helper method to plot both distance and theta lines.
        """
        self.distance_plot(distance_classified)
        self.theta_plot(theta_classified, line_width=1.0)

    def sample_plot(self, sample_pre, sample_post, circle_size=10.0):
        """Plots finitely-many sampled points.
        """
        colored_samples = [[] for i in range(5)]
        for inputs, label in zip(sample_pre, sample_post):
            x, y = self.attacker_position(inputs)
            colored_samples[label].append((x, y))
        for label, points in enumerate(colored_samples):
            plt.scatter([x for x, y in points], [y for x, y in points],
                        color=self.color(int(label)), s=circle_size)

    def single_line_plots(self, network_name, single_line_data):
        """Makes the single-line plots.
        """
        theta, distance, theta_sample, distance_sample = single_line_data

        self.theta_plot(theta)
        self.finalize_plot("%s/single_theta" % network_name, 2100, -5900)
        self.distance_plot(distance)
        self.finalize_plot("%s/single_distance" % network_name, 2100, -5900)
        self.sample_plot(*theta_sample, circle_size=20.0)
        self.finalize_plot("%s/single_sample_theta" % network_name,
                           2100, -5900)
        self.sample_plot(*distance_sample, circle_size=20.0)
        self.finalize_plot("%s/single_sample_distance" % network_name,
                           2100, -5900)

    def analyze(self):
        """Writes plots from the transformed line segments.
        """
        self.grayscale = (input("[G]rayscale or [C]olor? ").lower()[0] == "g")
        for i in range(1, 6):
            for j in range(1, 10):
                network_name = "acas_%d_%d" % (i, j)
                try:
                    distance_classified = self.read_artifact(
                        "%s/distance" % network_name)
                    theta_classified = self.read_artifact(
                        "%s/theta" % network_name)
                    sample_pre, sample_post = self.read_artifact(
                        "%s/sample" % network_name)
                    single_line_data = self.read_artifact(
                        "%s/single_lines" % network_name)
                except KeyError:
                    # Skip due to missing data.
                    continue
                print("Analyzing network:", network_name)
                self.distance_plot(distance_classified)
                self.finalize_plot("%s/distance" % network_name)
                self.theta_plot(theta_classified)
                self.finalize_plot("%s/theta" % network_name)
                self.overlapping_plot(distance_classified, theta_classified)
                self.finalize_plot("%s/overlapping" % network_name)
                self.sample_plot(sample_pre, sample_post)
                self.finalize_plot("%s/sample" % network_name)

                self.single_line_plots(network_name, single_line_data)
        return True

if __name__ == "__main__":
    ACASLinesExperiment("acas_lines").main()
