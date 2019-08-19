"""Methods for patching the ACAS Xu network with ExactLine.
"""
import numpy as np
from pysyrenn import NetPatcher, PlanesClassifier
from experiments.acas_planes import ACASPlanesExperiment
from polar_image import PolarImage

class NetPatchExperiment(ACASPlanesExperiment):
    """Experiment that attempts to apply 3 patches to the ACAS Xu network.

    NOTE: Region of interest is specified in experiment/acas_lines.py.
    """
    def spec_pockets(self, inputs):
        """Spec function for getting rid of the "pockets" of SR/SL.
        """
        labels = np.argmax(self.network.compute(inputs), axis=1)
        inputs = self.reset(inputs)
        velocity = inputs[0][-1]
        assert np.isclose(velocity, 150) or np.isclose(velocity, 200)
        is_generalized = np.isclose(velocity, 200)
        if not is_generalized:
            for i, point in enumerate(inputs):
                # Change strong right to weak right.
                if 1.0 < point[1] < 1.5 and 2750 < point[0] < 3750 and labels[i] == 4:
                    labels[i] = 2
                # Change strong left to weak left.
                if -1.5 < point[1] < -1.1 and 2250 < point[0] < 3500 and labels[i] == 3:
                    labels[i] = 1
        else:
            for i, point in enumerate(inputs):
                # Change strong right to weak right.
                if 0.94 < point[1] < 2.0 and 2750 < point[0] < 4500 and labels[i] == 4:
                    labels[i] = 2
                # Change strong left to weak left.
                if -2 < point[1] < -1.05 and 2250 < point[0] < 4500 and labels[i] == 3:
                    labels[i] = 1
        return labels

    def spec_bands(self, inputs):
        """Spec function for getting rid of the "back bands" behind the plane.
        """
        labels = np.argmax(self.network.compute(inputs), axis=1)
        inputs = self.reset(inputs)
        velocity = inputs[0][-1]
        assert np.isclose(velocity, 150) or np.isclose(velocity, 200)
        is_generalized = np.isclose(velocity, 200)
        # Fixes the "back-bands."
        if not is_generalized:
            for i, point in enumerate(inputs):
                if np.degrees(point[1]) > 0 and labels[i] == 1:
                    # Change weak left to weak right.
                    labels[i] = 2
        else:
            for i, point in enumerate(inputs):
                dist = point[0]
                rad = point[1]
                if (2.0 < rad < np.radians(+180) and
                        0 <= dist <= 5000 and
                        not (950 <= dist <= 1100 and 2 <= rad <= 2.55) and
                        labels[i] == 1):
                    # Change weak left to weak right.
                    labels[i] = 2
        return labels

    def spec_symmetry(self, inputs):
        """Spec function for making the SL/SR boundary symmetric.
        """
        labels = np.argmax(self.network.compute(inputs), axis=1)
        inputs = self.reset(inputs)
        for i, point in enumerate(inputs):
            if np.degrees(point[1]) > 0 and labels[i] == 3:
                labels[i] = 4
        return labels

    def run_for_spec(self, spec_name, spec_fn, layer_index, steps):
        """Patches the network for a given spec and stores the patched network.

        We also record:
        1. The spec for *both* the patching and generalizing regions.
        2. The polytopes from the SyReNN partitioning (as these are the same
            for the patched network) for *both* the patching and generalizing
            regions.
        3. The patched networks for all iterations of the patching algorithm.
        """
        # First, we use NetPatcher.from_spec_function to extract the desired
        # constraints for the patched region.
        patcher = NetPatcher.from_spec_function(
            self.network, layer_index, self.patch_region, spec_fn)
        key = "%s/spec" % spec_name
        patch_spec = (patcher.inputs, patcher.labels)
        self.record_artifact(patch_spec, key, "pickle")
        # Then get the SyReNN.
        syrenn = self.network.transform_plane(self.patch_region,
                                              compute_preimages=True,
                                              include_post=False)
        # We only need the preimages.
        key = "%s/syrenn" % spec_name
        self.record_artifact(syrenn, key, "pickle")

        # Then we do the same for the generalized region.
        generalized_patcher = NetPatcher.from_spec_function(
            self.network, layer_index,
            self.generalized_patch_region, spec_fn)
        generalized_patch_spec = (generalized_patcher.inputs,
                                  generalized_patcher.labels)
        key = "%s/spec_generalized" % spec_name
        self.record_artifact(generalized_patch_spec, key, "pickle")
        # Then get the SyReNN.
        syrenn = self.network.transform_plane(self.generalized_patch_region,
                                              compute_preimages=True,
                                              include_post=False)
        # We only need the preimages.
        key = "%s/syrenn_generalized" % spec_name
        self.record_artifact(syrenn, key, "pickle")

        # Now, we do the actual patching and record the patched networks.
        patcher.compute(steps=steps)
        data = self.begin_csv("%s/data" % spec_name, ["steps", "network", "time"])
        for step, patched_net in enumerate(patcher.intermediates):
            key = "%s/step_%03d" % (spec_name, step)
            self.record_artifact(patched_net, key, "masking_network")
            self.write_csv(data, {
                "steps": step,
                "network": key,
                "time": patcher.times[step]
            })

    def run(self):
        """Patches the network for a variety of specs and records the results.
        """
        # We only ever use one network for this Experiment, so we pre-load the
        # network, input helpers, and patch regions.
        self.network = self.load_network("acas_1_1")
        input_helpers = self.load_input_data("acas")
        self.process = input_helpers["process"]
        self.reset = input_helpers["reset"]

        self.patch_region = self.process(self.region_of_interest())

        self.own_velocity += 50
        self.intruder_velocity += 50
        self.generalized_patch_region = self.process(self.region_of_interest())

        specs = [
            ("spec_pockets", self.spec_pockets, 10),
            ("spec_bands", self.spec_bands, 10),
            ("spec_symmetry", self.spec_symmetry, 12)
        ][:int(input("Number of specs (1-3): "))]
        steps = int(input("Number of steps (per spec): "))
        self.record_artifact(list(zip(*specs))[0], "specs", "pickle")
        for spec_name, spec_fn, layer in specs:
            self.run_for_spec(spec_name, spec_fn, layer, steps)

    def plot_from_pre_syrenn(self, pre_syrenn, network, reset, key):
        """Given the SyReNN partitioning of an ACAS network, plot it.

        Note that @pre_syrenn should be *just* the partitioning, i.e.
        polytopes, not the post-images.
        """
        post_syrenn = list(map(network.compute, pre_syrenn))
        syrenn = list(zip(pre_syrenn, post_syrenn))
        classifier = PlanesClassifier.from_syrenn([syrenn])

        polytopes, labels = classifier.compute()[0]
        polytopes = [reset(polytope)[:, :2] for polytope in polytopes]
        color_labels = list(map(self.color, labels))

        max_rho = np.sqrt(max(self.min_x**2, self.max_x**2) +
                          max(self.min_y**2, self.max_y**2))

        polar_plot = PolarImage((2*max_rho, 2*max_rho), (1001, 1001))
        polar_plot.plot_polygons(polytopes, color_labels, 30)
        polar_plot.circle_frame(max_rho, "#ffffff")
        self.record_artifact(polar_plot.image, key, "rgb_image")

    def compute_percent_met(self, network, spec):
        """Computes the percent of @spec is met by @network.

        @spec should be a tuple (processed-inputs, corresponding-labels).
        """
        inputs, labels = spec
        real_labels = np.argmax(network.compute(inputs), axis=1)
        return np.count_nonzero(labels == real_labels) / len(labels)

    def analyze(self):
        """Plots the patched networks and records constraints met.
        """
        specs = self.read_artifact("specs")
        reset = self.load_input_data("acas")["reset"]
        for spec in specs:
            patch_met = self.begin_csv("%s/patch_met" % spec,
                                       ["Step", "Percent Met"])
            patch_spec = self.read_artifact("%s/spec" % spec)
            patch_syrenn = self.read_artifact("%s/syrenn" % spec)

            gen_met = self.begin_csv("%s/gen_met" % spec,
                                     ["Step", "Percent Met"])
            gen_spec = self.read_artifact("%s/spec_generalized" % spec)
            gen_syrenn = self.read_artifact(
                "%s/syrenn_generalized" % spec)

            data = self.read_csv("%s/data" % spec)
            for step_data in data:
                steps = int(step_data["steps"])
                network = self.read_artifact(step_data["network"])
                print("Analyzing for step:", steps)
                # First, compute how many of the specs are met.
                pct_met = self.compute_percent_met(network, patch_spec)
                self.write_csv(patch_met,
                               {"Step": steps, "Percent Met": pct_met})
                pct_met = self.compute_percent_met(network, gen_spec)
                self.write_csv(gen_met,
                               {"Step": steps, "Percent Met": pct_met})
                # Then, plot them.
                self.plot_from_pre_syrenn(patch_syrenn, network, reset,
                                          "%s/patch_%03d" % (spec, steps))
                self.plot_from_pre_syrenn(gen_syrenn, network, reset,
                                          "%s/gen_%03d" % (spec, steps))
        return True

if __name__ == "__main__":
    NetPatchExperiment("netpatch").main()
