"""Abstract descriptions of a network analyzer.
"""
import base64
import csv
import io
import os
import pickle
import shutil
import tarfile
import numpy as np
from PIL import Image
import imageio
from pysyrenn import Network
from syrenn_proto import syrenn_pb2 as syrenn_pb

class Experiment:
    """Abstract class describing a network experiment.
    """
    def __init__(self, directory_name):
        """Initializes a new experiment.

        Creates an output directory, removing any existing files in that
        location. Also initializes a new artifacts_csv file which holds a list
        of all the artifacts written to using the "record_artifact" interface.
        """
        # Create a directory outside of bazel-bin for storing the results.
        global_dir = os.environ["BUILD_WORKING_DIRECTORY"]
        self.directory = "{}/experiments/results/{}".format(
            global_dir, directory_name)
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)

        self.tar_name = "%s.exp.tgz" % self.directory
        self.open_files = []
        self.artifacts = None
        self.artifacts_csv = self.begin_csv("artifacts", ["key", "type", "path"])

    def close(self, tar=True, nicely=True):
        """Ends the experiment, freeing open file pointers.

        @tar determines whether the experiment directory should be tarred into
        an archive. In general, this is done after the initial experiments and
        then once more if the analysis produces any new files.

        @nicely should indicate whether the closing is expected or not. For
        example, if the program errors in the middle of an experiment, it is
        not "nice." nicely=False will leave the experiment directory alone
        (i.e. untarred and unremoved).
        """
        for open_file in self.open_files:
            open_file.close()
        self.open_files = []

        if tar and nicely:
            # tar directory into directory.exp.tar
            with tarfile.open(self.tar_name, "w:gz") as archive:
                for name in os.listdir(self.directory):
                    archive.add("%s/%s" % (self.directory, name), arcname=name)
        if nicely:
            shutil.rmtree(self.directory, ignore_errors=True)

    def open(self):
        """Reads experiment data from a previous run.

        In general, this is called after run() and close(), or when doing an
        analyze-only execution of previous experimental results.
        """
        # Create the extraction directory.
        shutil.rmtree(self.directory, ignore_errors=True)
        os.mkdir(self.directory)

        # Extract the tar file.
        with tarfile.open(self.tar_name, "r:*") as archive:
            archive.extractall(self.directory)

        self.artifacts = self.read_csv("artifacts")
        # Re-open and re-fill the CSV file so we can keep writing to it.
        self.artifacts_csv = self.begin_csv("artifacts", ["key", "type", "path"])
        # TODO(masotoud): look into a way to open the file for appending.
        # instead of truncating + re-adding.
        for artifact in self.artifacts:
            self.write_csv(self.artifacts_csv, artifact)

    def has_archive(self):
        """True if the experiment seems to have already been run.
        """
        return os.path.exists(self.tar_name)

    def remove_archive(self):
        """Removes an existing archive.
        """
        return os.remove(self.tar_name)

    def __del__(self):
        """Close file handles in case of unexpected exit.

        Normal exits should call .close(nicely=True).
        """
        self.close(nicely=False)

    @staticmethod
    def image_to_datauri(image):
        """Converts a Numpy array holding an image to a data representation.

        Useful for embedding images in SVG files. See:
        https://stackoverflow.com/questions/46598607
        """
        image = Image.fromarray(image.astype("uint8"))
        raw_bytes = io.BytesIO()
        image.save(raw_bytes, "PNG")
        raw_bytes.seek(0)
        image = base64.b64encode(raw_bytes.read()).decode()
        return "data:image/png;base64,{}".format(image)

    @staticmethod
    def load_network(network_name, maxify_acas=True):
        """Loads an experiment network given by @network_name.

        Currently supports models of the form:
        - acas_#_# (ACAS Xu models translated from the ReluPlex format)
        - {cifar10,mnist}_relu_#_# (fully-connected ReLU models from ERAN)
        - {cifar10,mnist}_relu_conv{small,medium,big}{_diffai,_pgd}
          (convolutional ReLU models from ERAN).

        And should be referenced in BUILD rule experiments:models.

        maxify_acas controlls whether the ACAS model is "cleaned" before
        returned; cleaning removes the unnecessary ReLU layer at the end as
        well as inverts the outputs so the recommended action becomes the
        maximal score.
        """
        if "acas_" in network_name:
            _, i, j = network_name.split("_")
            network = Network.from_file("models/acas_models/%s_%s.eran"
                                        % (i, j))
            if maxify_acas:
                # We remove ReLU layers from the end of the model as they don't
                # actually change the classification (when one exists).
                assert not hasattr(network.layers[:-1], "weights")
                network.layers = network.layers[:-1]

                # ACAS Xu networks use the minimal score as the class instead
                # of the more-standard maximum score; this inverts the last
                # layer so the minimal score becomes the max.
                network.layers[-1].weights *= -1.0
                network.layers[-1].biases *= -1.0
            return network
        if "vrl_" in network_name:
            _, model_name = network_name.split("_", 1)
            return Network.from_file("models/vrl/eran/%s.eran" % model_name)
        return Network.from_file(
            "external/%s_model/file/model.eran" % (network_name))

    @staticmethod
    def load_input_data(name_or_path, is_eran_conv_model=False):
        """Gets a dataset and/or its metadata.

        Currently supports three datasets:
        - acas (empty dataset which returns preprocessing info for ACAS)
        - cifar10_test (100 test images from ERAN)
        - mnist_test (100 test images from ERAN)

        Returns a dictionary with four items:
        - process(np_array) will process a raw (uint8) Numpy array image into a
          format that can be passed to the Network.
        - reset(np_array) will invert process(...). This may not always be
          possible if process(...) is non-invertible, but it should at least
          work on all valid images (i.e., uint8 pixel values).
        - raw_inputs holds (flattened) uint8 Numpy arrays for each input image.
        - labels holds the corresponding label for each input image.
        """
        if name_or_path == "acas":
            mins = np.array([0.0, -3.141593, -3.141593, 100.0, 0.0])
            maxes = np.array([60760.0, 3.141593, 3.141593, 1200.0, 1200.0])
            means = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
            std_deviations = np.array([60261.0, 6.28318530718, 6.28318530718,
                                       1100.0, 1200.0])
            return {
                "process": lambda i: ((np.clip(i, mins, maxes) - means) / std_deviations),
                "reset": lambda i: ((i * std_deviations) + means),
                "raw_inputs": [],
                "labels": [],
            }
        inputs_file_path = "external/%s_data/file/data.csv" % name_or_path
        # ERAN models
        with open(inputs_file_path, "r", newline="") as inputs_file:
            csv_inputs = csv.reader(inputs_file)
            input_data = np.array(list(csv_inputs)).astype(np.float64)

        # TODO(masotoud): handle this more robustly.
        process_input = lambda i: i / 255.0
        reset_input = lambda i: np.round(i * 255.0)
        if "cifar10" in name_or_path and not is_eran_conv_model:
            # The "conv" ERAN models correspond to .pyt files, which are
            # normalized *in the model itself* (see:
            # https://github.com/eth-sri/eran/blob/df4d1cf4556c0ad4ba062cbfda9a645449f8096e/tf_verify/__main__.py
            # line 254 and 263 -- the normalize() call for Pytorch models
            # corresponds to a NormalizeLayer that we pass to the transformer
            # server)
            process_input = lambda i: (i / 255.0) - 0.5
            reset_input = lambda i: np.round((i + 0.5) * 255.0)
        return {
            "process": process_input,
            "reset": reset_input,
            "raw_inputs": input_data[:, 1:],
            "labels": input_data[:, 0].astype(np.int),
        }

    @staticmethod
    def rgbify_image(image):
        """Converts a flattened uint8 image array into a HWC, RGB one.

        This method *ASSUMES* the dataset is either CIFAR10 or MNIST.
        """
        shape_to_use = (-1,)
        if image.size == 28 * 28:
            # We assume this is a B/W MNIST image
            shape_to_use = (28, 28)
        elif image.size == (32 * 32 * 3):
            # We assume this is an RGB CIFAR10 32x32 image
            shape_to_use = (32, 32, 3)
        image = np.reshape(image, shape_to_use).astype(np.uint8)
        if len(shape_to_use) == 2:
            # https://stackoverflow.com/questions/32171917
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        return image

    def begin_csv(self, filename, column_labels, extrasaction="raise"):
        """Opens a new CSV file with the given column labels for writing.

        Returns a tuple (file_handle, csv_writer) that can be passed to
        write_csv. These do not need to be manually flushed or closed --- that
        is handled by Experiment.close() and Experiment.write_csv().

        @filename should be a path-safe identifier for the CSV file (extension
            and path not necessary).
        @column_labels should be a list of (string) column labels. These will
            correspond to dictionary keys in write_csv and read_csv.
        """
        dirname = os.path.dirname(filename)
        self.artifact_directory(dirname) # Ensure the directory exists
        csv_file = open("%s/%s.csv" % (self.directory, filename), "w",
                        newline="")
        csv_writer = csv.DictWriter(csv_file, column_labels,
                                    extrasaction=extrasaction)
        csv_writer.writeheader()
        self.open_files.append(csv_file)
        return (csv_file, csv_writer)

    @staticmethod
    def write_csv(csv_data, record):
        """Writes a record to a CSV file opened with Experiment.begin_csv(...).

        @csv_data should be the tuple returned by Experiment.begin_csv(...)
        @record should be a dictionary with keys corresponding to the
            @column_labels passed to Experiment.begin_csv(...)
        """
        csv_data[1].writerow(record)
        csv_data[0].flush()

    def read_csv(self, filename):
        """Fully reads a CSV file and returns a list of the rows.

        Each row is represented by a dictionary with keys corresponding to the
        columns. Dictionary values are strings --- parsing them to a usable
        format is left to the caller.
        """
        filename = "%s/%s.csv" % (self.directory, filename)
        with open(filename, "r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = []
            for record in csv_reader:
                data.append(dict(record))
        return data

    def artifact_directory(self, dir_key):
        """Creates a directory that will be included in the experiment archive.

        Returns its path without trailing /.
        """
        name = "%s/%s" % (self.directory, dir_key)
        os.makedirs(name, exist_ok=True)
        return name

    def record_artifact(self, artifact, key, artifact_type):
        """Record a high-level artifact from the experiment.

        Each Experiment instance has a corresponding "artifact store" which
        allows one to easily record, store, and later reference artifacts
        produced during the experiment. This method adds an artifact @artifact
        to that store, using key @key under the assumption that the artifact
        should be treated as type @artifact_type.
        """
        filename = "%s/%s" % (self.directory, key)

        file_directory = os.path.dirname(filename)
        if artifact_type != "rawpath" and not os.path.exists(file_directory):
            # See notes on a possible race condition in the answer here:
            # https://stackoverflow.com/questions/10149263
            os.makedirs(file_directory)

        def write_pb(path, pb_serialized):
            """Writes @pb_serialized to @path.
            """
            with open(path, "wb") as to_file:
                to_file.write(pb_serialized.SerializeToString())

        if artifact_type == "rgb_image":
            filename += ".png"
            imageio.imwrite(filename, artifact)
        elif artifact_type == "np_array":
            filename += ".npy"
            np.save(filename, artifact)
        elif artifact_type == "pickle":
            filename += ".pickle"
            with open(filename, "wb") as to_file:
                pickle.dump(artifact, to_file)
        elif artifact_type == "matplotlib":
            filename += ".pdf"
            artifact.savefig(filename)
        elif artifact_type in "network":
            filename += ".pb"
            write_pb(filename, artifact.serialize())
        elif artifact_type == "svg":
            filename += ".svg"
            artifact.saveas(filename)
        elif artifact_type == "rawpath":
            filename = artifact
        elif artifact_type == "csv":
            filename = artifact
        elif artifact_type == "text":
            with open(filename, "w") as to_file:
                to_file.write(artifact)
        else:
            raise NotImplementedError
        record = {"key": key, "type": artifact_type, "path": filename}
        self.write_csv(self.artifacts_csv, record)
        if self.artifacts is not None:
            self.artifacts.append(record)

    def read_artifact(self, key):
        """Reads an artifact from the loaded artifact store indexed by @key.

        Experiment.open() *MUST* be called before using read_artifact(...).
        This method is intended to be used only by the analyze() method (not
        run, which should be calling record_artifact).
        """
        assert self.artifacts is not None
        try:
            artifact = next(artifact for artifact in self.artifacts
                            if artifact["key"] == key)
        except StopIteration:
            raise KeyError

        def read_pb(path, pb_type):
            """Deserializes protobuf data stored to a file.

            @path is the file path, @pb_type is the Protobuf descriptor to
            parse as.
            """
            with open(path, "rb") as from_file:
                string_rep = from_file.read()
            serialized = pb_type()
            serialized.ParseFromString(string_rep)
            return serialized

        if artifact["type"] == "rgb_image":
            return imageio.imread(artifact["path"])
        if artifact["type"] == "np_array":
            return np.load(artifact["path"], allow_pickle=True)
        if artifact["type"] == "pickle":
            with open(artifact["path"], "rb") as from_file:
                return pickle.load(from_file)
        if artifact["type"] == "rawpath":
            return artifact["path"]
        if artifact["type"] == "csv":
            return self.read_csv(artifact["path"])
        if artifact["type"] == "network":
            return Network.deserialize(
                read_pb(artifact["path"], syrenn_pb.Network))
        raise NotImplementedError

    @staticmethod
    def summarize(data):
        """Returns a string summarizing the contents of @data.

        @data should be a one-dimensional Numpy array.
        """
        return "Mean: %lf, Std: %lf, Mdn: %lf, 25%%: %lf, 75%%: %lf" % (
            np.mean(data),
            np.std(data),
            np.median(data),
            np.percentile(data, 25.0),
            np.percentile(data, 75.0))

    def run(self):
        """Runs the analysis on the network and inputs.
        """
        raise NotImplementedError

    def analyze(self):
        """Performs analysis and summarization after a run().

        Experiment.read_artifact(key) should be used to recover data from the
        experiment.
        """
        raise NotImplementedError

    def main(self):
        """Main experiment harness.
        """
        run = not self.has_archive()
        if not run:
            print("It seems that this experiment has already been run.")
            choice = input("[R]e-run, [A]nalyze-only, or [D]elete and re-run? ").lower()[0]
            assert choice in {"r", "a", "d"}
            if choice == "d":
                self.remove_archive()
            run = choice in {"r", "d"}
        if run:
            self.run()
            self.close()
        self.open()
        did_modify = self.analyze()
        self.close(tar=did_modify)
