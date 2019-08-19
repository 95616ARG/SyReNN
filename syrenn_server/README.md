# SyReNN Server
This contains C++ code that computes SyReNN for 1- and 2-dimensional input
restriction domains of interest. At a high-level, we have a few main types of
files:

1. ``server.cc`` implements the GRPC server that handles receiving queries from
   the clients and dispatching calls to the rest of the C++ methods. Methods of
   particular interest include:
   - ``main``, which registers all of the layers supported by the server and
     sets the address to use.
   - ``transform_polytope``, which computes the symbolic representation for a
     particular network (series of layers) and input UPolytope (see below).
   - ``transform_lines`` does the same for a set of ``SegmentedLine``
     instances. It uses ``split_line`` to dynamically sub-divide the line and
     avoid running out of memory.
2. ``{layer}_transformer.{h, cc}`` contain algorithms to compute SyReNN for
   certain layer types. Currently, only 1- and 2-dimensional input regions are
   supported for the non-linear transformers.
    1. ``pwl_transformer.{h, cc}`` contains algorithms for computing SyReNN for
       any piecewise-linear function with convex-polytope partitions. It is
       sub-classed by, eg., ``relu_transformer.{h, cc}`` and
       ``argmax_transformer.{h, cc}``.
    2. ``affine_transformer.{h, cc}`` contains algorithms for computing SyReNN
       for any affine function. It is sub-classed by, eg.,
       ``conv2d_transformer.{h, cc}`` and ``fullyconnected_transformer.{h,
       cc}``.
3. ``strided_window_data.{h, cc}`` contains a helper class for all layers that
   work with striding windows (eg. 2D Convolution and MaxPool).
4. ``segmented_line.{h, cc}`` and ``upolytope.{h, cc}`` are the underlying
   datatypes used to represent SyReNN's input and output for 1- and
   arbitrarily-many dimensions (respectively). See below for more details.
5. ``shared.h`` contains a few helpful typedefs; in particular, it helps to
   know that ``RMMatrixXf`` stands for "**R**ow-**M**ajor Eigen **Matrix** in
   arbitrary (**X**) dimensions using single-precision **f**loats."

# Core Data Structures
We use two core data structures, described below.

## SegmentedLine
The ``SegmentedLine`` class represents, unsurprisingly, a segmented line ---
i.e., a union of connected line segments in potentially-high-dimensional space.
We represent the line as a vector of endpoints.

Each endpoint has two main representations, a _preimage ratio_ and a
_post-value_ (usually referred to just as "endpoint" in the code). Using the
terminology of [1], the preimage ratios correspond to the actual ExactLine
partitioning, while the post-values are the result of applying the network to
the point described by the corresponding preimage ratio. Note that each
preimage ratio is between 0.0 and 1.0, with 0.0 being the start point of the
line before applying any layers and 1.0 being the endpoint before applying any
layers.

Furthermore, there is an associated class ``SegmentedLineStub`` which is like a
``SegmentedLine`` except it only has preimage ratios. This is used by the
server to dynamically sub-divide lines, and is very effective in controlling
the memory usage on large networks. The basic idea is that a very large
``SegmentedLine`` can be broken up into a smaller ``SegmentedLine`` (say, maybe
only the first 10 endpoints) with the rest of the endpoints being stored in a
memory-friendly ``SegmentedLineStub`` until the first 10 endpoints have been
fully transformed.

To actually perform transformations (compute ExactLine), however, the
``SegmentedLineStub`` needs to be turned back into a ``SegmentedLine``, which
necessitates re-computing the post-values. To help control this computational
cost, we keep track of the _layer at which each endpoint was introduced_,
allowing us to wait until that layer is encountered in the re-computing process
before adding the endpoint back. This is kept track of primarily by the member
``interpolate_before_layer_``.

Finally, we note that transforming a line with a layer is actually a two step
process: first, endpoints are added to the line (just their preimage ratios,
not their post-values); then, the corresponding postimage-values are computed.
We separate these two steps, as the postimage-values are often extremely
memory-intensive, so we need to decide how to sub-divide the line (to save
memory, see above) before they are actually computed (but after we know how
many there are). This also has the slight benefit that we do not _have_ to
actually compute the very last layer's post-values if the client only wants the
partitioning in the input space.

## UPolytope
The ``UPolytope`` class represents a _union of convex polytopes_. Note that
this is a generalization of the ``SegmentedLine`` concept to higher-dimensional
regions (the existence of many important line-specific optimizations motivates
our keeping two separate classes).

Each convex polytope is represented by its vertices, i.e. in a V-representation
(which is efficient and precise for low-dimensional regions). Notably, when the
layers used are continuous, many of the polytopes tend to share vertices.
``UPolytope`` handles this by keeping a "master-list" of all vertices used by
polytopes in the union, then each polytope keeps a vector of indices into that
master list to specify what its own vertices are.

Each endpoint again has two representations, the preimage _combination_ and
post-value. These have the same interpretation as in a ``SegmentedLine``, with
the slight difference that the preimage combination has one entry for each
vertex in the pre-transformation ``UPolytope`` instead of just one ratio (this
is a natural extension of a ratio-along-a-line for higher-dimensional regions).

Notably, we have not made as many optimizations for memory usage with
``UPolytope`` as we did with ``SegmentedLine``. This is due primarily to the
use cases we have explored for each; image networks with extremely large
activation vectors for ``SegmentedLine`` vs. relatively smaller controller
networks for ``UPolytope``.

Instead, we have optimized for parallelization: we use TBB concurrency-safe
containers, and have a two-phased vertex-insertion process that minimizes the
number of expensive Eigen resizes we need to do. This is primarily documented
in ``upolytope.h`` and example uses are in ``pwl_transformer.cc``.
