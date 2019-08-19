"""Methods for plotting polar polygons in a PNG.
"""
import numpy as np
import tqdm

class PolarImage:
    """Polar coordinates plotter.
    """
    def __init__(self, plot_shape, png_shape, plot_origin=None, silent=False):
        """Initalizes a new PolarImage with the given shapes.

        @plot_shape should be the (Cartesian) bounds on the data to be plotted.
        @png_shape should be the shape of the PNG in (height, width)-pixels.
        @plot_origin should be (Cartesian) coordinate to place at the center of
            the PNG image. If None (the default), the center of the plot region
            (@plot_shape) will be used.
        """
        self.image = np.full(tuple(png_shape) + (3,), 255).astype(np.uint8)
        self.plot_shape = plot_shape
        self.png_shape = png_shape
        if plot_origin is None:
            plot_origin = [self.plot_shape[0] / 2, self.plot_shape[1] / 2]
        self.plot_origin = plot_origin
        self.silent = silent

    def place_rgba(self, image, png_center):
        """Places an RGBA image @image on the plot centered at @png_center.
        """
        png_center_y, png_center_x = png_center
        image_height, image_width = image.shape[:2]
        png_start_y = png_center_y - (image_height // 2)
        png_start_x = png_center_x - (image_width // 2)

        alphas = np.expand_dims(image[:, :, 3], 3).astype(np.float32) / 255.0
        existing = self.image[png_start_y:(png_start_y + image_height),
                              png_start_x:(png_start_x + image_width)]
        new = ((1.0 - alphas) * existing) + (alphas * image[:, :, :3])
        self.image[png_start_y:(png_start_y + image_height),
                   png_start_x:(png_start_x + image_width)] = new

    @staticmethod
    def hex_to_int(hex_color):
        """Converts a hex color to an integer array.

        Only supports strings of the form "#123456"
        """
        hex_color = hex_color.replace("#", "")
        red = int(hex_color[:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
        return [red, green, blue]

    def plot_to_png(self, plot_y, plot_x):
        """Converts a plot-coordinate to a png-coordinate (pixel).

        Returns (png_y, png_x) with the idea that self.image[png_y, png_x] is
        the pixel corresponding to Cartesian point (plot_x, plot_y).

        NOTE: The inputs are (plot_y, plot_x), not (plot_x, plot_y).
        NOTE: The return value is a float; the caller should decide how to
            round to the nearest pixel.
        NOTE: Plot y-coordinates are ordered bottom-to-top while PNG
            y-coordinates are ordered top-to-bottom!
        """
        plot_height, plot_width = self.plot_shape
        png_height, png_width = self.png_shape

        plot_y += self.plot_origin[0]
        plot_x += self.plot_origin[1]

        png_y = plot_y * (png_height / plot_height)
        png_y = (png_height - 1) - png_y

        png_x = plot_x * (png_width / plot_width)
        return (png_y, png_x)

    def png_to_plot(self, png_y, png_x):
        """Converts a png-coordinate (pixel) to a plot-coordinate.

        This is the inverse of @plot_to_png.
        NOTE: Plot y-coordinates are ordered bottom-to-top while PNG
            y-coordinates are ordered top-to-bottom!
        """
        plot_height, plot_width = self.plot_shape
        png_height, png_width = self.png_shape

        plot_y = (png_height - 1) - png_y
        plot_y = plot_y * (plot_height / png_height)

        plot_x = png_x * (plot_width / png_width)
        return (plot_y - self.plot_origin[0], plot_x - self.plot_origin[1])

    @staticmethod
    def max_cosine(start, end):
        """Returns max(cos(theta)) for -pi <= start <= theta <= end <= pi.
        """
        assert start <= end
        if start <= 0.0 <= end:
            return +1
        return max(np.cos(start), np.cos(end))

    @staticmethod
    def min_cosine(start, end):
        """Returns min(cos(theta)) for -pi <= start <= theta <= end <= pi.
        """
        return min(np.cos(start), np.cos(end))

    @staticmethod
    def max_sine(start, end):
        """Returns max(sin(theta)) for -pi <= start <= theta <= end <= pi.
        """
        if start <= (np.pi / 2.0) <= end:
            return +1.0
        return max(np.sin(start), np.sin(end))

    @staticmethod
    def min_sine(start, end):
        """Returns min(sin(theta)) for -pi <= start <= theta <= end <= pi.
        """
        assert start <= end
        if start <= (-np.pi / 2.0) <= end:
            return -1.0
        return min(np.sin(start), np.sin(end))

    @classmethod
    def polar_cartesian_box(cls, vertices):
        """Given a *polar*-polytope, return a *Cartesian* box containing it.

        Our main algorithm (below) basically guess-and-checks each pixel, so we
        use this to limit the region of guessing.

        The basic idea is to decompose the problem into finding Cartesian boxes
        around each of the edges, then joining them together.
        """
        if len(vertices) > 2:
            global_x_box = [np.Infinity, -np.Infinity]
            global_y_box = [np.Infinity, -np.Infinity]
            # Decompose into edges and join back up.
            for start, end in zip(vertices[:-1], vertices[1:]):
                x_box, y_box = cls.polar_cartesian_box([start, end])
                global_x_box[0] = min(x_box[0], global_x_box[0])
                global_y_box[0] = min(y_box[0], global_y_box[0])
                global_x_box[1] = max(x_box[1], global_x_box[1])
                global_y_box[1] = max(y_box[1], global_y_box[1])
            return global_x_box, global_y_box
        # This is just a line segment. We effectively find the min/max rho and
        # min/max cos/sin on the segment. Multiplying the two gives us safe
        # upper/lower bound.
        start, end = vertices
        start_rho, start_theta = start
        end_rho, end_theta = end
        min_rho, max_rho = sorted((start_rho, end_rho))
        min_theta, max_theta = sorted((start_theta, end_theta))
        max_cos = cls.max_cosine(min_theta, max_theta)
        min_cos = cls.min_cosine(min_theta, max_theta)
        max_sin = cls.max_sine(min_theta, max_theta)
        min_sin = cls.min_sine(min_theta, max_theta)
        return np.array([
            [min(np.floor((min_rho * min_cos, max_rho * min_cos))),
             max(np.ceil((min_rho * max_cos, max_rho * max_cos)))],
            [min(np.floor((min_rho * min_sin, max_rho * min_sin))),
             max(np.ceil((min_rho * max_sin, max_rho * max_sin)))]])

    @staticmethod
    def polygon_contains(polygon, point):
        """True if @point is inside of @polygon.

        NOTE: This uses code from softSurfer, see below this class for
        reference. @polygon should be in V-representation (i.e., a Numpy array
        of counter-clockwise vertices).
        """
        return polyline_contains(polygon, point)

    def plot_polygon(self, box, polygon, color):
        """Plots a @polygon given its Cartesian @box, and corresponding @color.

        @box should be computed with .polar_cartesian_box(@polygon).
        @polygon should be a Numpy array of counter-clockwise vertices
            (V-representation polytope) describing a polygon in Polar
            space. Note that I think _technically_ any polygon would work, but
            I haven't tested it.
        @color should be a string hex color to plot, compatible with
            .hex_to_int.
        """
        x_box, y_box = box

        png_y_start, png_x_start = self.plot_to_png(y_box[0], x_box[0])
        png_y_start, png_x_start = int(png_y_start), int(png_x_start)

        png_y_end, png_x_end = self.plot_to_png(y_box[1], x_box[1])
        png_y_end, png_x_end = int(np.ceil(png_y_end)), int(np.ceil(png_x_end))

        # These get inverted when we switch to PNG.
        png_y_start, png_y_end = sorted((png_y_start, png_y_end))
        png_y_start = max(png_y_start - 1, 0)
        png_y_end = min(png_y_end + 1, self.image.shape[0])
        png_x_start = max(png_x_start - 1, 0)
        png_x_end = min(png_x_end + 1, self.image.shape[1])
        color = self.hex_to_int(color)
        for png_y in range(png_y_start, png_y_end):
            if np.array_equiv(self.image[png_y, png_x_start:png_x_end, :],
                              color):
                continue
            for png_x in range(png_x_start, png_x_end):
                plot_y, plot_x = self.png_to_plot(png_y, png_x)
                rho = np.linalg.norm((plot_x, plot_y))
                theta = np.arctan2(plot_y, plot_x)
                if self.polygon_contains(polygon, [rho, theta]):
                    self.image[png_y, png_x, :] = color

    def window_plot(self, polygons, colors, n_splits):
        """Plots @polygons when possible by quantizing the space.

        The basic idea is that, in many plots, there exist Cartesian boxes
        ("windows") that contain many polygons and all of the same color. This
        method plots (a subset of) those boxes by:
        1. Slide a window over the PNG image; for each window:
            1a. Find (a superset of) all polygons that overlap with the window,
                along with their corresponding colors.
            1b. If they all have the same color, plot the window as that color
                and continue to 1c.  Otherwise, go on to the next window.
            1c. Find (a subset of) all polygons that lie entirely within the
                window, and delete them and their corresponding colors from
                @polygons and @colors.

        The super/subset parenthesized remarks above refer to the fact that we
        over-approximate each polytope with a box to make the steps feasible.
        However, when including the super/subset, everything still holds
        correctly.

        We use tuples of (start, end) for intervals and tuples of (y_interval,
        x_interval) for boxes.
        """
        def interval_overlaps(interval1, interval2):
            """True if interval1 has an intersection with interval2.
            """
            return not (interval1[1] < interval2[0] or
                        interval2[1] < interval1[0])
        def box_overlaps(box1, box2):
            """True if box1 has an overlap with box2.
            """
            return (interval_overlaps(box1[0], box2[0]) and
                    interval_overlaps(box1[1], box2[1]))
        def interval_contains(big, small):
            """True if interval "big" entirely contains interval "small."
            """
            return big[0] <= small[0] and small[1] <= big[1]
        def box_contains(big, small):
            """True if box "big" entirely contains box "small."
            """
            return (interval_contains(big[0], small[0]) and
                    interval_contains(big[1], small[1]))
        # Precompute the boxes of all @polygons; after this, we won't use
        # @polygons at all, just the boxes.
        boxes = list(map(self.polar_cartesian_box, polygons))
        y_step = self.image.shape[0] // n_splits
        x_step = self.image.shape[1] // n_splits
        for y_start in range(0, self.image.shape[0], y_step):
            for x_start in range(0, self.image.shape[0], x_step):
                y_min, x_min = self.png_to_plot(y_start, x_start)
                y_end = min(y_start + y_step, self.image.shape[0])
                x_end = min(x_start + x_step, self.image.shape[1])
                y_max, x_max = self.png_to_plot(y_end, x_end)
                # Note that png_to_plot flips the order of y_coordinates
                # (because PNG pixels are indexed top-to-bottom instead of
                # bottom-to-top).
                window_box = [(x_min, x_max), sorted((y_min, y_max))]
                # (1a) Find all overlapping polytopes.
                overlapping_i = [i for i, box in enumerate(boxes)
                                 if box_overlaps(window_box, box)]
                window_colors = set(colors[i] for i in overlapping_i)
                if len(window_colors) != 1:
                    # (1b) Multiple polytopes (possibly) in this window have
                    # different colors --- we can't safely plot this window in
                    # a single color.
                    continue
                # (1b) They all have the same color, so plot it!
                window_color = self.hex_to_int(next(iter(window_colors)))
                self.image[y_start:y_end, x_start:x_end, :] = window_color

                # (1c) Remove entirely-contained polytopes.
                contained_i = [i for i in overlapping_i
                               if box_contains(window_box, boxes[i])]
                for i in contained_i[::-1]:
                    del polygons[i]
                    del boxes[i]
                    del colors[i]

    def circle_frame(self, rho, color):
        """Plots a circular frame around plot-(0, 0) with min-radius @rho.

        The circular frame includes the @rho-circle about the origin and all
        circles with radius greater that @rho.

        We use this because .window_plot will "color outside the lines," so we
        need to clean its mess up before getting a final plot.

        The implementation essentially looks for the intersection points in
        plot-coordinates then converts them to png-coordinates and corrects
        them.

        NOTE: This method is untested and probably shouldn't be used in cases
        where the @rho-circle extends beyond the bounds of the plot.
        """
        for png_y in range(self.image.shape[0]):
            plot_y, _ = self.png_to_plot(png_y, 0)
            # Now we need to find where the rho-circle intersects with y =
            # plot_y. We have y = rho*sin(theta), so the intersection is
            # theta = arcsin(y / rho).
            theta = np.arcsin(plot_y / rho)
            plot_x_intersection = rho * np.abs(np.cos(theta))
            _, png_x_intersection = self.plot_to_png(0, -plot_x_intersection)
            png_x_intersection = int(png_x_intersection)
            self.image[png_y, 0:png_x_intersection, :] = self.hex_to_int(color)
            _, png_x_intersection = self.plot_to_png(0, +plot_x_intersection)
            png_x_intersection = int(png_x_intersection)
            self.image[png_y, png_x_intersection:, :] = self.hex_to_int(color)

    def plot_polygons(self, polygons, colors, plot_windows=0):
        """Plots a set of polar polygons on the image.

        @polygons should be a list of Numpy arrays, each one a set of vertices
            for a particular polygon with shape (n_vertices, 2={rho, theta})
        @colors should be a list of string hex colors (prepended with "#")
            associated with each polygon in @polygons.
        @plot_windows controls whether a first-pass plotting is performed. If
            @plot_windows <= 0, no first-pass plotting is performed. If
            @plot_windows > 0, then @plot_windows**2 windows will be used for
            the first-pass plotting. Note that, while this can significantly
            speed up the plotting, it may color pixels outside of @polygons, so
            two considerations should be made:
            1. @plot_windows should *NOT* be used if you intend to call
               @plot_polygons multiple times.
            2. Points not covered by @polygons may be colored in when
                @plot_windows is used; if necessary, you should manually clear
                such points using, eg., .circle_frame if @polygons form a
                circle about 0.0 (i.e., partition a rectangle in polar-space).
        """
        if plot_windows > 0:
            # window_plot modifies polygons/colors, lets make sure "the buck
            # stops here" (so callers don't have to worry about it).
            polygons = polygons.copy()
            colors = colors.copy()
            self.window_plot(polygons, colors, plot_windows)
        boxes = list(map(self.polar_cartesian_box, polygons))
        box_sizes = [(box[0][1] - box[0][0])*(box[1][1] - box[1][0])
                     for box in boxes]
        ordered = sorted(range(len(boxes)), key=lambda i: box_sizes[i],
                         reverse=True)
        for i in tqdm.tqdm(ordered, disable=self.silent):
            self.plot_polygon(boxes[i], polygons[i], colors[i])

# Copyright 2001, softSurfer (www.softsurfer.com)
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.

# Translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>.
# https://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py

# is_left(): tests if a point is Left|On|Right of an infinite line.
#   Input: three points P0, P1, and P2
#   Return: >0 for P2 left of the line through P0 and P1
#           =0 for P2 on the line
#           <0 for P2 right of the line
#   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])
def polyline_contains(V, P):
    wn = 0   # the winding number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] < P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn != 0
