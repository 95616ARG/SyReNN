"""Methods using ExactLine to understand classification of a network on a line.
"""
import numpy as np
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.argmax_layer import ArgMaxLayer

class LinesClassifier:
    """Handles classifying a set of lines using ExactLine.
    """
    def __init__(self, network, lines, preimages=True):
        """Creates a new LinesClassifier for the given @network and @lines.

        @lines should be a list of (startpoint, endpoint) tuples. If
        preimages=True is set, preimages of the endpoints of each
        classification region will be returned (otherwise, only the ratio
        between startpoint and endpoint will be).
        """
        self.network = network
        self.lines = lines
        self.preimages = preimages

        self.partially_computed = False
        self.transformed_lines = None

        self.computed = False
        self.classifications = None

    def partial_compute(self):
        """Computes the relevant ExactLine and stores it for analysis.
        """
        if self.partially_computed:
            return

        self.transformed_lines = self.network.exactlines(
            self.lines, compute_preimages=self.preimages, include_post=True)
        self.partially_computed = True

    @classmethod
    def from_exactlines(cls, transformed_lines):
        """Constructs a partially-computed LinesClassifier from ExactLines.

        This is useful, for example, if you need ExactLines for some other
        analysis and then want to determine classification regions. We use it
        to determine the class break-point when generating Figure 4 from [1]
        (experiments/linearity_hypothesis.py).
        """
        if not len(transformed_lines[0]) == 2:
            error = ("ExactLine must be called with include_post=True " +
                     "to use from_exactline.")
            if len(transformed_lines) == 2:
                error += ("\nIf you called exactline (singular), you must " +
                          "pass a singleton list instead.")
            raise TypeError(error)

        self = cls(None, None, None)
        self.transformed_lines = transformed_lines
        self.partially_computed = True
        return self

    def compute(self):
        """Returns the classification regions of network restricted to line.

        Returns a list with one tuple (pre_regions, corresponding_labels) for
        each line in self.lines. pre_regions is a list of tuples of endpoints
        that partition each input line.
        """
        if self.computed:
            return self.classifications

        self.partial_compute()

        self.classifications = []
        classify_network = Network([ArgMaxLayer()])
        for pre, post in self.transformed_lines:
            # First, we take each of the linear regions and split them where
            # the ArgMax changes.
            lines = list(zip(post[:-1], post[1:]))
            classify_transformed_lines = classify_network.exactlines(
                lines, compute_preimages=False, include_post=False)

            split_pre = []
            split_post = []
            for i, endpoints in enumerate(classify_transformed_lines):
                pre_delta = pre[i + 1] - pre[i]
                post_delta = post[i + 1] - post[i]
                for point_ratio in endpoints:
                    point_pre = pre[i] + (point_ratio * pre_delta)
                    point_post = post[i] + (point_ratio * post_delta)
                    if i == 0 or not point_ratio == 0.0:
                        split_pre.append(point_pre)
                        split_post.append(point_post)

            # Now, in each of the resulting regions, we compute the
            # corresponding label.
            region_labels = []
            for i in range(len(split_pre) - 1):
                mid_post = 0.5 * (split_post[i] + split_post[i + 1])
                region_labels.append(np.argmax(mid_post))

            # Finally, we merge segments with the same classification.
            merged_pre = []
            merged_labels = []
            for i, label in enumerate(region_labels):
                if not merged_labels or label != merged_labels[-1]:
                    merged_pre.append(split_pre[i])
                    merged_labels.append(label)
            merged_pre.append(split_pre[-1])
            regions = list(zip(merged_pre[:-1], merged_pre[1:]))
            self.classifications.append((regions, merged_labels))
        self.computed = True
        return self.classifications
