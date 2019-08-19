"""Methods for interfacing with the VRL models.

Used by model_checking.py and
../../third_party/reluplex_model_checking/export_specs.py
"""
import numpy as np
import pypoman

class VRLModel:
    """Class representing one of the models from the VRL project.

    Models are from the project: https://github.com/caffett/VRL_CodeReview
    """
    def __init__(self, model_name):
        """Creates a new VRLModel.

        @model_name should be one of:
        {"pendulum_continuous", "satelite", "quadcopter"}
        """
        self.model_name = model_name

    @staticmethod
    def box_to_constraints(bounds):
        """Converts a set of (uniform) bounds to H-representation.

        Returns A_ub such that a point x is in the box defined by bounds[0] <=
        x[0] <= bounds[1] and bounds[0] <= x[0] <= bounds[1] if (and only if)
        A_ub*x <= 0.
        """
        return np.array([
            [1, 0, -bounds[1]],
            [0, 1, -bounds[1]],
            [-1, 0, bounds[0]],
            [0, -1, bounds[0]],
        ])

    @staticmethod
    def box_to_vertices(bounds):
        """Extracts a V-representation from a box.

        @bounds should be a tuple (low, high), the box is taken to be:
        { (x, y) | low <= x <= high ^ low <= y <= high }
        """
        low, high = bounds
        return np.array([
            [low, low],
            [high, low],
            [high, high],
            [low, high],
        ])

    def init_set(self, as_box=False, as_vertices=False):
        """Returns the set of initial states for the model.

        This should be a subset of safe_set.
        """
        box = {
            "pendulum_continuous": [-0.35, 0.35],
            "satelite": [-1.0, 1.0],
            "quadcopter": [-0.5, 0.5],
        }[self.model_name]
        if as_box:
            return box
        if as_vertices:
            return self.box_to_vertices(box)
        return self.box_to_constraints(box)

    def safe_set(self, as_box=False):
        """Returns the set of safe states for the model.
        """
        box = {
            "pendulum_continuous": [-0.5, 0.5],
            "satelite": [-1.5, 1.5],
            "quadcopter": [-1.0, 1.0],
        }[self.model_name]
        if as_box:
            return box
        return self.box_to_constraints(box)

    def env_transition(self):
        """Returns matrices describing the transition function for the model.

        Returns a tuple (tA, tB). For old state x and action a, the new state
        x' is defined to be: x' := x + ((tA . x) + (tB . a)).

        Note that, in contrast to the VRL repository, here we roll the scale
        and the timestep into the transition matrices themselves. The end
        result is equivalent while being less complex (IMO).
        """
        return {
            "pendulum_continuous":
                (0.01 * np.array([[0., 1.], [10.0/1.0, 0.]]),
                 0.01 * 15.0 *
                 np.array([[0.], [1.0]])),
            "satelite":
                (0.01 * np.array([[2., -1.], [1, 0.]]),
                 0.01 * 10.0 *
                 np.array([[2.0], [0.]])),
            "quadcopter":
                (0.01 * np.array([[1., 1.], [0., 1.]]),
                 0.01 * 15.0 *
                 np.array([[0.], [1.0]])),
        }[self.model_name]

    def env_step(self, pre_state, action):
        """Performs one step of the model given a pre-state and action.

        See env_transition above for the meaning of A, B.
        """
        A, B = self.env_transition()
        delta = (np.matmul(A, pre_state) + np.matmul(B, action)).reshape(-1)
        return np.asarray(pre_state + delta).squeeze()

    def hole_set(self):
        """Returns the set of safe states guaranteed to map back into init.

        hole_set = { x | x in safe ^ step(x) in init_set }

        This is returned in an H-Representation. Points in the hole_set are
        guaranteed to be safe for n steps as long as all points in init_set but
        outside the hole_set are safe for n steps. Thus, we can ignore these
        points (or any inputs that get mapped into hole_set) when performing
        BMC.
        """
        hole_A = []
        transition_A, transition_B = self.env_transition()
        if np.any(transition_B < 0):
            raise NotImplementedError
        init_box = self.init_set(as_box=True)
        # This assumes that the model has a HardTanh after it, so its results
        # are within +-1:
        # new_state_bounds = old_state + tA.old_state +- tB)
        # max_new_state = (old_state + tA.old_state) + tB
        # min_new_state = (old_state + tA.old_state) - tB
        A_ub = transition_A + np.eye(2)
        # Add the upper bounds.
        for dim in range(2):
            tB = transition_B.flatten()[dim]
            new_constraint = [A_ub[dim, 0], A_ub[dim, 1],
                              tB - init_box[1]]
            hole_A.append(new_constraint)
        # Add the lower bounds.
        for dim in range(2):
            tB = transition_B.flatten()[dim]
            # ... + b >= init[0]
            # -... - b <= -init[0]
            # -... - b + init[0] <= 0
            new_constraint = [-A_ub[dim, 0], -A_ub[dim, 1],
                              init_box[0] + tB]
            hole_A.append(new_constraint)
        return np.array(hole_A)

    def disjunctive_safe_set(self):
        """Set of points in the safe_set that are not in the hole_set.

        See the comment in hole_set --- these are the only points we "actually"
        need to care about.
        """
        hole_A_ub = self.hole_set()
        safe_A_ub = self.safe_set(as_box=False)
        planes = []
        for hole_face in hole_A_ub:
            appended = np.append(safe_A_ub, [-hole_face], axis=0)
            vertices = pypoman.polygon.compute_polygon_hull(appended[:, :-1],
                                                            -appended[:, -1])
            planes.append(np.array(vertices))
            safe_A_ub = np.append(safe_A_ub, [hole_face], axis=0)
        return planes
