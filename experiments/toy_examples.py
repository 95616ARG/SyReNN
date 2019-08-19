"""Methods for producing the toy example figure from [2].
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from pysyrenn import ReluLayer, FullyConnectedLayer, Network, PlanesClassifier
from pysyrenn import MaskingNetwork
from experiments.acas_planes import ACASPlanesExperiment

class ToyExamplesExperiment(ACASPlanesExperiment):
    """Experiment for producing the toy example figures from [2].
    """
    grayscale = False

    def color(self, label):
        """Returns a color to use for a class region on the overview example.
        """
        if self.grayscale:
            return ("#ffffff", "#222222")[label]
        return ("#555555", "#88dd88")[label]

    def patching_color(self, label):
        """Returns a color to use for a class region on the NetPatch example.

        We use blue to differentiate from the habitability network.
        """
        if self.grayscale:
            return ("#ffffff", "#222222")[label]
        return ("#555555", "#8888dd")[label]

    def partition_color(self, region):
        """Returns a color to use for a particular linear partition.

        Grayscale not yet supported.
        """
        assert not self.grayscale
        return "C%d" % region

    @classmethod
    def habitability_network(cls, params=False):
        """Returns the habitability network from the overview.

        If @params=True, returns a list of the parameters of the network. This
        option is used to linearize the network around a point in .linearize(),
        which is in turn used to explicitly state the maps in LaTeX.
        """
        A1 = np.array([[-1.0, 0.25, 1],
                       [+1.0, 0.5, 1],
                       [0, 1, 0],
                       [0.5, 0.5, 2]]).T
        b1 = np.array([1, -1, -1, -5])
        A2 = np.array([[-2, 1.0, 1.0, 1],
                       [1.0, 2.0, -1.0, 2]]).T
        b2 = np.array([1, 0])
        if params:
            return [A1, b1, A2, b2]
        return Network([FullyConnectedLayer(A1, b1),
                        ReluLayer(),
                        FullyConnectedLayer(A2, b2)])

    @classmethod
    def patchable_network(cls):
        """Returns the network used in the patching section.
        """
        A1 = np.array([[-1.,  1.],
                       [ 1.,  0.],
                       [ 0.,  1.]]).T
        b1 = np.array([-0.5, 0., 0.])
        A2 = np.array([[1.,  1., 1.],
                       [0., -1., -1.]]).T
        b2 = np.array([0., 1.])
        return Network([FullyConnectedLayer(A1, b1),
                        ReluLayer(),
                        FullyConnectedLayer(A2, b2)])

    @classmethod
    def patched_network(cls):
        """Returns the patched network used in the patching section.
        """
        activation_layers = cls.patchable_network().layers
        A1 = activation_layers[0].weights.numpy().copy()
        b1 = activation_layers[0].biases.numpy().copy()
        A1[0, 0] = 0.0
        patched_layer = FullyConnectedLayer(A1, b1)
        value_layers = [patched_layer] + activation_layers[1:]
        return MaskingNetwork(activation_layers, value_layers)

    def run_habitability(self):
        """Records data related to the habitability example network.
        """
        network = self.habitability_network()
        self.record_artifact(network, "habitability/network", "network")
        # We use an x/y box with fixed z for our region of interest.
        x = [0, 3]
        y = [0, 3]
        z = 0
        interest_region = np.array([[x[0], y[0], z],
                                    [x[1], y[0], z],
                                    [x[1], y[1], z],
                                    [x[0], y[1], z]])
        # We want preimages (for plotting) & postimages (for labeling).
        syrenn = network.transform_plane(interest_region, True, True)
        self.record_artifact(syrenn, "habitability/syrenn", "pickle")
        classifier = PlanesClassifier.from_syrenn([syrenn])
        class_regions = classifier.compute()[0]
        self.record_artifact(class_regions, "habitability/class_regions",
                             "pickle")

    def run_patching(self):
        """Records data related to the patching example network.
        """
        network = self.patchable_network()
        self.record_artifact(network, "patchable/network", "network")
        # We use an x/y box with fixed z for our region of interest.
        x = [-3, 3]
        y = [-3, 3]
        interest_region = np.array([[x[0], y[0]],
                                    [x[1], y[0]],
                                    [x[1], y[1]],
                                    [x[0], y[1]]])
        # We want preimages (for plotting) & postimages (for labeling).
        syrenn = network.transform_plane(interest_region, True, True)
        self.record_artifact(syrenn, "patchable/syrenn", "pickle")
        classifier = PlanesClassifier.from_syrenn([syrenn])
        class_regions = classifier.compute()[0]
        self.record_artifact(class_regions, "patchable/class_regions",
                             "pickle")
        # Now the patched network.
        patched = self.patched_network()
        patched_syrenn = []
        for pre, post in syrenn:
            patched_syrenn.append((pre, patched.compute(pre)))
        classifier = PlanesClassifier.from_syrenn([patched_syrenn])
        class_regions = classifier.compute()[0]
        self.record_artifact(class_regions, "patched/class_regions", "pickle")

    def run(self):
        """Saves the network, SyReNN, and classification for later analysis.
        """
        self.run_habitability()
        self.run_patching()

    @classmethod
    def linearize(cls, point):
        """Returns the linear map corresponding to the region around @point.
        """
        A1, b1, A2, b2 = cls.habitability_network(params=True)
        first_layer_output = np.matmul(np.array([point]), A1) + b1
        mask = np.maximum(np.sign(first_layer_output), 0.0)
        linear = mask * A1
        linear = np.matmul(linear, A2)
        offset = np.matmul((mask * b1), A2)
        offset += b2
        return linear, offset

    @staticmethod
    def plot_borders(polygons):
        """Plots the borders of @polygons in thick white.

        Used to overlay the linear regions on the class plot.
        """
        for polygon in polygons:
            poly = patches.Polygon(polygon[:, :2], fill=False,
                                   edgecolor="white", linewidth=3.0)
            plt.gca().add_patch(poly)

    def analyze_habitability(self):
        """Plots the habitability network and outputs the symbolic representation.
        """
        def latexify(vertex):
            """Returns a LaTeX format of a point.
            """
            return "({}, {}, {})".format(*vertex)
        def latexify_matrix(matrix):
            """Returns a LaTeX format of a matrix.
            """
            latex = "\\begin{bmatrix} "
            for row in matrix:
                latex += " & ".join(map(str, row)) + "\\\\ "
            latex += "\\end{bmatrix}"
            return latex

        syrenn = self.read_artifact("habitability/syrenn")

        latex = []
        for pre, post in syrenn:
            unique_vertices = np.unique(pre, axis=0)
            if len(unique_vertices) == 1:
                continue
            affine_map, bias = self.linearize(np.mean(unique_vertices, axis=0))
            line = "(\\Hull(\\{ " + ", ".join(map(latexify, unique_vertices)) + " \\}), x \\mapsto "
            line += latexify_matrix(affine_map) + "x + "
            line += latexify_matrix(bias) + "),"
            latex.append(line)

        latex = "\n".join(latex)
        self.record_artifact(latex, "habitability/hatr.tex", "text")

        syrenn_regions = [pre for pre, post in syrenn]

        self.plot_as_cartesian((syrenn_regions, range(len(syrenn_regions))),
                               self.partition_color)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        self.record_artifact(plt, "habitability/linear_partitions",
                             "matplotlib")
        plt.clf()

        class_regions = self.read_artifact("habitability/class_regions")
        self.plot_as_cartesian(class_regions)
        self.plot_borders(syrenn_regions)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        self.record_artifact(plt, "habitability/class_regions", "matplotlib")
        plt.clf()

    def analyze_patching(self):
        """Plots the patching example network and patch specification.
        """
        syrenn = self.read_artifact("patchable/syrenn")
        syrenn_regions = [pre for pre, post in syrenn]

        class_regions = self.read_artifact("patchable/class_regions")
        self.plot_as_cartesian(class_regions, self.patching_color)
        self.plot_borders(syrenn_regions)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        self.record_artifact(plt, "patchable/class_regions", "matplotlib")

        self.plot_as_cartesian(
            ([np.array([[-3, -3], [0.5, -3.], [0.5, 0], [0, 0.5], [-3, 0.5]]),
              np.array([[-3., 0.5], [3., 0.5], [3., 3.], [-3., 3.]]),
              np.array([[0., 0.5], [0.5, 0.], [3., 0.], [3., 0.5]]),
              np.array([[0., 0.5], [0.5, 0.], [3., 0.], [3., 0.5]]),
              np.array([[0.5, -3.], [3., -3.], [3., 0.], [0.5, 0.]]),
             ], list(range(5))),
            lambda i: "C{}".format(i + 1),
            0.8)
        plt.text(-1.5, -1.5, "P1", fontsize=16)
        plt.text(-0.15, 2, "P2", fontsize=16)
        plt.text(1, 0.1, "P3", fontsize=16)
        plt.text(1.5, -1.5, "P4", fontsize=16)
        self.record_artifact(plt, "patchable/patch_spec", "matplotlib")
        plt.clf()

        class_regions = self.read_artifact("patched/class_regions")
        self.plot_as_cartesian(class_regions, self.patching_color)
        self.plot_borders(syrenn_regions)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        self.record_artifact(plt, "patched/class_regions", "matplotlib")
        plt.clf()

    def analyze(self):
        """Prints the SyReNN in LaTeX format and makes the plots.
        """
        self.analyze_habitability()
        self.analyze_patching()
        return True

if __name__ == "__main__":
    ToyExamplesExperiment("toy_examples").main()
