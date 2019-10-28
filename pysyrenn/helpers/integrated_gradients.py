"""Methods using ExactLine to exactly compute Integrated Gradients.
"""
import numpy as np
import gc

class IntegratedGradients:
    """Class to orchestrate computation of Integrated Gradients.
    """
    def __init__(self, network, lines, batch_size=1024):
        """Initializes a new Integrated Gradients computer class.

        @batch_size is the maximum number of points to compute gradients for at
        a time, used to control memory usage.
        """
        self.network = network
        self.lines = lines
        self.batch_size = batch_size

        self.partially_computed = False
        self.exactlines = None
        self.n_samples = None

        self.attributions = dict()

    def partial_compute(self):
        """Computes the sampling regions needed to get an exact IG computation.
        """
        if self.partially_computed:
            return

        self.exactlines = self.network.exactlines(
            self.lines, compute_preimages=False, include_post=False)
        self.n_samples = [len(endpoints) - 1 for endpoints in self.exactlines]
        self.partially_computed = True

    def compute_attributions(self, label):
        """Computes IG attributions for output label @label.
        """
        if label in self.attributions:
            return self.attributions[label]

        self.partial_compute()

        self.attributions[label] = []
        for i, (start, end) in enumerate(self.lines):
            delta = end - start
            endpoints = self.exactlines[i]

            attributions = np.zeros_like(start)
            for batch_start in range(0, self.n_samples[i], self.batch_size):
                batch_end = batch_start + self.batch_size
                batch_endpoints = endpoints[batch_start:batch_end]

                sample_points = (batch_endpoints[:-1] + batch_endpoints[1:])
                sample_points /= 2.0
                sample_points = start + np.outer(sample_points, delta)

                gradients = self.network.compute_gradients(sample_points,
                                                           label)
                for i, region_gradient in enumerate(gradients):
                    region_start, region_end = batch_endpoints[i:(i + 2)]
                    region_size = region_end - region_start
                    attributions += region_size * region_gradient
                del gradients
                gc.collect()
            attributions *= delta
            self.attributions[label].append(attributions)
        return self.attributions[label]
