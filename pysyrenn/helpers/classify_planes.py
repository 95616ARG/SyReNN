"""Methods using SyReNN to understand classification of a network on a plane.
"""
import numpy as np
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.argmax_layer import ArgMaxLayer

class PlanesClassifier:
    """Handles classifying a set of planes using SyReNN.
    """
    def __init__(self, network, planes, preimages=True):
        """Creates a new PlanesClassifier for the given @network and @planes.

        @planes should be a list of Numpy arrays with each one representing a
        V-representation polytope with (n_vertices, n_dims). If preimages=True
        is set, preimages of the endpoints of each classification region will
        be returned (otherwise, only the combinations will be).
        """
        self.network = network
        self.planes = planes
        self.preimages = preimages

        self.partially_computed = False
        self.transformed_planes = None

        self.computed = False
        self.classifications = None

    def partial_compute(self):
        """Computes the relevant ExactLine and stores it for analysis.
        """
        if self.partially_computed:
            return

        self.transformed_planes = self.network.transform_planes(self.planes,
                                                                self.preimages,
                                                                True)
        self.partially_computed = True

    @classmethod
    def from_syrenn(cls, transformed_planes):
        """Constructs a partially-computed PlanesClassifier from ExactLines.
        """
        self = cls(None, None, None)
        self.transformed_planes = transformed_planes
        self.partially_computed = True
        return self

    def compute(self):
        """Returns the classification regions of network restricted to @planes.

        Returns a list with one tuple (pre_regions, corresponding_labels) for
        each plane in self.planes. pre_regions is a list of Numpy arrays, each
        one representing a VPolytope.

        In contrast to LinesClassifier, no attempt is made here to return the
        minimal set.
        """
        if self.computed:
            return self.classifications

        self.partial_compute()

        self.classifications = []
        classify_network = Network([ArgMaxLayer()])
        for upolytope in self.transformed_planes:
            pre_polytopes = []
            labels = []
            # First, we take each of the linear partitions and split them where
            # the ArgMax changes.
            postimages = [post for pre, post in upolytope]
            classified_posts = classify_network.transform_planes(
                postimages, compute_preimages=False, include_post=False)
            for vpolytope, classify_upolytope in zip(upolytope,
                                                     classified_posts):
                pre, post = vpolytope
                for combinations in classify_upolytope:
                    pre_polytopes.append(np.matmul(combinations, pre))

                    mean_combination = np.mean(combinations, axis=0)
                    class_region_label = np.argmax(
                        np.matmul(mean_combination, post).flatten())
                    labels.append(class_region_label)
            self.classifications.append((pre_polytopes, labels))
        self.computed = True
        return self.classifications
