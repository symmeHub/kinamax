import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
from jax import lax, vmap
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from diffrax import PIDController
from typing import NamedTuple, ClassVar
from collections import namedtuple, defaultdict

try:
    import cuml
except ImportError:
    pass
from sklearn.cluster import DBSCAN, AgglomerativeClustering


@dataclass
class Container:
    """
    Container class for holding various parameters.
    """

    def as_dict(self, flatten=True):
        """
        Converts the problem to a dictionary.
        Returns:
            dict: Dictionary representation of the problem.
        """
        raw_dic = self.__dict__
        dic = {}
        for key, value in raw_dic.items():
            if flatten:
                v = value.flatten()
            else:
                v = value
            dic[key] = np.array(v)
        return dic

    def as_polars(self, repeat=0):
        """
        Converts the problem to a Polars DataFrame.
        Returns:
            pl.DataFrame: Polars DataFrame representation of the problem.
        """
        dic = self.as_dict()
        # keys = list(dic.keys())
        # s = len(dic[keys[0]])
        # label = np.arange(s)
        # label_name = self.label_col if hasattr(self, "label_col") else "label"
        # dic[label_name] = label
        if repeat > 0:
            for key, value in dic.items():
                dic[key] = value.repeat(repeat)
        df = pl.DataFrame(dic)
        return df


@register_dataclass
@dataclass
class AttractorFinderConfig(Container):
    init_time: jax.Array = field(default_factory=lambda: jnp.array(0.0, dtype=float))
    init_time_step: jax.Array = field(
        default_factory=lambda: jnp.array(1.0e-3, dtype=float)
    )
    convergence_tol: jax.Array = field(
        default_factory=lambda: jnp.array(1.0e-6, dtype=float)
    )
    target_frequency: float = 50.0
    subharmonic_factor: float = 10.0
    # label_col: ClassVar = "config_id"


def convert_subharmonics_flags(subharmonics_flags, final_flags, targetted_subharmonics):
    """
    Convert the detected subharmonics into a boolean array.
    """
    detected_subharmonics = np.zeros(final_flags.shape, dtype=np.int32)
    for i in range(len(subharmonics_flags)):
        sh = subharmonics_flags[i]
        if final_flags[i] == 1:
            detected_subharmonics[i] = targetted_subharmonics[i, sh == 1][0]
    return detected_subharmonics


@register_dataclass
@dataclass
class AttractorFinderSolution(Container):
    """
    Solution of the orbit finder.
    """

    attractors: jax.Array
    detected_subharmonic: jax.Array
    subharmonic_residual: jax.Array
    minimum_residual: jax.Array
    simulated_periods: jax.Array
    simulated_time: jax.Array
    final_flag: jax.Array
    simulated_iterations: jax.Array
    converged: jax.Array

    def as_dict(self, state_vector_labels=None):
        """
        Converts the problem to a dictionary.
        Returns:
            dict: Dictionary representation of the problem.
        """
        raw_dic = self.__dict__
        dic = {}
        s = self.detected_subharmonic.shape
        orbit = np.arange(np.prod(s[:-1]))[..., None].repeat(s[-1]).flatten()
        attractor = (
            np.arange(s[-1])[:, None].repeat(np.prod(s[:-1]), axis=1).T.flatten()
        )
        dic["sim_label"] = orbit
        dic["attractor_label"] = attractor
        for key, value in raw_dic.items():
            value = np.array(value)
            if key == "attractors":
                Nv = value.shape[-1]
                value = value.reshape(-1, Nv)
                if state_vector_labels is not None:
                    for i in range(Nv):
                        dic[state_vector_labels[i]] = value[:, i]
                else:
                    for i in range(Nv):
                        dic[f"Xa_{i}"] = value[:, i]
            else:
                dic[key] = value.flatten()
        return dic

    def get_subharmonics(self):
        """
        Returns the detected subharmonics.
        Returns:
            jnp.ndarray: Detected subharmonics.
        """
        subharmonics_flags = self.subharmonics_flags
        final_flags = self.final_flags
        targetted_subharmonics = self.targetted_subharmonics
        Nts = targetted_subharmonics.shape[-1]
        subharmonics_flags = subharmonics_flags.reshape(-1, Nts)
        final_flags = final_flags.flatten()
        targetted_subharmonics = targetted_subharmonics.reshape(-1, Nts)
        return convert_subharmonics_flags(
            subharmonics_flags, final_flags, targetted_subharmonics
        )

    def as_polars(self, *args, **kwargs):
        """
        Converts the problem to a Polars DataFrame.
        Returns:
            pl.DataFrame: Polars DataFrame representation of the problem.
        """
        dic = self.as_dict(*args, **kwargs)
        # orbit = dic.pop("orbit_label", None)

        df = pl.DataFrame(dic)
        # df.insert_column(0, pl.Series("sim_label", orbit))
        return df


class AttractorFinder(NamedTuple):
    """
    Container class for finding orbits in a given ODE problem.
    """

    residuals_per_period: jax.Array = np.array(10, int)
    targetted_subharmonics: jax.Array = np.array([1, 3, 5], int)
    max_periods: int = 1000
    controller: PIDController = PIDController(rtol=1e-7, atol=1e-9)
    solver: Tsit5 = Tsit5()

    def get_max_subharmonic(self):
        """
        Returns the maximum subharmonic.
        Returns:
            int: Maximum subharmonic.
        """
        return np.max(self.targetted_subharmonics)

    def get_time_steps_number(self):
        """
        Returns the number of time steps.
        Returns:
            int: Number of time steps.
        """
        max_subharmonic = self.get_max_subharmonic()
        return 2 * max_subharmonic * self.residuals_per_period + 1

    def get_max_shooting_iterations(self):
        """
        Returns the maximum number of shooting iterations.
        Returns:
            int: Maximum number of shooting iterations.
        """
        max_subharmonic = self.get_max_subharmonic()
        return self.max_periods // (2 * max_subharmonic)

    def find_attractors(
        self,
        problem: NamedTuple,
        init_conditions: jax.Array,
        finder_config: AttractorFinderConfig,
    ):
        """
        Find orbits for the given problem.
        Args:
            problem: The problem instance.
            term (ODETerm): The ODE term.
            solver (Tsit5): The solver.
            controller (PIDController): The controller.
            init_conditions (jax.Array): Initial conditions for the system.
            target_frequency (float): Target frequency.
            finder_config (AttractorFinderConfig): Configuration for finding orbits.
        Returns:
            jax.Array: Found orbits.
        """

        def body_fun(carry):
            """
            Body of the while loop that integrates the ODE and checks for convergence.
            """

            start_time = carry.start_time
            end_time = carry.end_time
            init_conditions = carry.init_conditions
            iteration = carry.iteration
            finder_config = carry.finder_config
            convergence_tol = finder_config.convergence_tol
            flag = carry.flag
            # max_shooting_iterations = finder_config.get_max_shooting_iterations()
            (
                Xout,
                res,
                start_time,
                end_time,
            ) = AttractorFinder.integrate_and_check_convergence(
                init_conditions=init_conditions,
                start_time=start_time,
                end_time=end_time,
                steps_number=time_steps_number,
                problem=carry.problem,
                solver=solver,
                controller=controller,
                init_time_step=carry.finder_config.init_time_step,
                # target_frequency=carry.target_frequency,
                targetted_subharmonics=targetted_subharmonics,
                residuals_per_period=residuals_per_period,
            )
            iteration += 1
            flag = jnp.where(
                jnp.any(res <= convergence_tol), 1, flag
            )  # CONVERGENCE ACHIEVED
            flag = jnp.where(
                iteration >= max_shooting_iterations, 2, flag
            )  # EXCEED MAX ITERATIONS
            carry = Carry(
                iteration=iteration,
                flag=flag,
                residuals=res,
                start_time=start_time,
                end_time=end_time,
                init_conditions=Xout,
                problem=carry.problem,
                target_frequency=carry.target_frequency,
                finder_config=carry.finder_config,
            )
            return carry

        def condition_fun(carry):
            """
            Stop condition for the while loop. The loop continues until convergence is achieved
            or the maximum number of iterations is reached.
            """

            flag = carry.flag
            return jnp.all(flag == 0)

        Carry = namedtuple(
            "Carry",
            [
                "iteration",
                "residuals",
                "start_time",
                "end_time",
                "init_conditions",
                "problem",
                "target_frequency",
                "finder_config",
                "flag",
            ],
        )
        solver = self.solver
        controller = self.controller
        residuals_per_period = self.residuals_per_period
        targetted_subharmonics = self.targetted_subharmonics
        target_frequency = finder_config.target_frequency
        max_subharmonic = self.get_max_subharmonic()
        time_steps_number = self.get_time_steps_number()
        max_shooting_iterations = self.get_max_shooting_iterations()
        init_time = finder_config.init_time
        subharmonic_factor = finder_config.subharmonic_factor
        target_period = 1.0 / target_frequency
        duration = 2.0 * max_subharmonic * target_period
        start_time = finder_config.init_time
        end_time = start_time + duration
        iteration = jnp.array(0)
        flag = jnp.array(0)
        residuals = jnp.zeros(targetted_subharmonics.shape)

        carry_in = Carry(
            iteration=iteration,
            flag=flag,
            residuals=residuals,
            start_time=start_time,
            end_time=end_time,
            init_conditions=init_conditions,
            problem=problem,
            target_frequency=target_frequency,
            finder_config=finder_config,
        )
        carry_out = lax.while_loop(
            body_fun=body_fun,
            cond_fun=condition_fun,
            init_val=carry_in,
        )
        final_flag = carry_out.flag
        end_time = carry_out.start_time
        final_conditions = carry_out.init_conditions
        simulated_time = end_time - init_time
        iterations = carry_out.iteration
        simulated_periods = 2 * max_subharmonic * iterations
        residuals = carry_out.residuals
        subharmonic_flag = (
            (final_flag == 1) & (residuals <= residuals.min() * subharmonic_factor)
        ) * 1
        # subharmonic_flag = ((final_flag == 1) & (residuals == residuals.min())) * 1
        # subharmonic_mask = targetted_subharmonics.max() + 1
        detected_subharmonic = jnp.where(
            subharmonic_flag == 1, targetted_subharmonics, jnp.inf
        ).min()
        detected_subharmonic = jnp.where(
            detected_subharmonic == jnp.inf, 0, detected_subharmonic
        ).astype(int)
        subharmonic_residual = jnp.where(
            detected_subharmonic == targetted_subharmonics, residuals, jnp.inf
        ).min()
        min_residual = residuals.min()
        # CALCULATE ATTRACTORS
        term = ODETerm(problem.rhs)
        tg = jnp.arange(1, max_subharmonic + 1)
        t0 = finder_config.init_time
        t1 = t0 + max_subharmonic * target_period
        dt0 = finder_config.init_time_step
        ts = tg * target_period + t0
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            final_conditions,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )

        attractors = sol.ys

        solution = AttractorFinderSolution(
            attractors=attractors,
            detected_subharmonic=detected_subharmonic
            * np.ones(max_subharmonic, dtype=int),
            subharmonic_residual=subharmonic_residual
            * np.ones(max_subharmonic, dtype=float),
            minimum_residual=min_residual * np.ones(max_subharmonic, dtype=float),
            simulated_periods=simulated_periods * np.ones(max_subharmonic, dtype=int),
            simulated_time=simulated_time * np.ones(max_subharmonic, dtype=float),
            converged=(final_flag == 1) * np.ones(max_subharmonic, dtype=bool),
            final_flag=final_flag * np.ones(max_subharmonic, dtype=int),
            simulated_iterations=iterations * np.ones(max_subharmonic, dtype=int),
        )
        return problem, finder_config, init_conditions, solution

    @staticmethod
    def calculate_subharmonic_atomic_residual(pos, args):
        """
        Calculates the shooting residual for a given subharmonic and a given time step.
        """
        norm, offset, X, state_weights = args
        residual = (X[-pos - offset - 1] - X[-pos - 1]) * state_weights
        norm += (residual * residual).sum()
        return norm, offset, X, state_weights

    @staticmethod
    def calculate_subharmonic_residual(
        subharmonic, X, residuals_per_period, state_weights
    ):
        """
        Calculates the shooting residual for a given subharmonic and at all time steps.
        """
        calculate_residuals = AttractorFinder.calculate_subharmonic_atomic_residual
        offset = residuals_per_period * subharmonic
        args_in = (0.0, offset, X, state_weights)
        out = lax.fori_loop(0, offset, calculate_residuals, args_in)
        return out[0] * subharmonic**2 / residuals_per_period

    @staticmethod
    def integrate_and_check_convergence(
        init_conditions,
        start_time,
        end_time,
        steps_number,
        problem,
        solver,
        controller,
        init_time_step,
        targetted_subharmonics,
        residuals_per_period,
    ):
        """
        Integrate the ODE between t0 and t1, and check the convergence of the subharmonic residuals.
        """
        t0 = start_time
        t1 = end_time
        dt0 = init_time_step
        X0 = init_conditions
        term = ODETerm(problem.rhs)
        # Vectorized version of calculate_subharmonic_residual
        batched_calculate_subharmonic_residual = vmap(
            AttractorFinder.calculate_subharmonic_residual,
            in_axes=(0, None, None, None),
        )
        tg = jnp.arange(steps_number)
        ts = tg * (t1 - t0) / (steps_number - 1) + t0
        ts = ts.at[0].set(t0)  # Ensure the first time step
        ts = ts.at[-1].set(t1)  # Ensure the last time step
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            X0,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )

        Xs = sol.ys
        state_weights = problem.state_weights()
        res = batched_calculate_subharmonic_residual(
            targetted_subharmonics, Xs, residuals_per_period, state_weights
        )
        X2 = Xs[-1]
        t2 = t1 + (t1 - t0)
        return X2, res, t1, t2


def post_process_attractor_finder_results(
    problem_class,
    problems,
    finder_configs,
    init_conditions,
    solutions,
    target_subharmonics,
    solution_state_labels,
):
    """Flatten batched results and balance rows per subharmonic."""

    state_vector_labels = problem_class.state_vector_labels
    state_space_dim = len(state_vector_labels)
    flattened_init = np.array(init_conditions).reshape(-1, state_space_dim)
    max_attractors = target_subharmonics.max()

    df_init_conditions = pl.DataFrame(
        {
            k: flattened_init[:, i].repeat(max_attractors)
            for i, k in enumerate(state_vector_labels)
        }
    )

    raw_data = pl.concat(
        [
            problems.as_polars(repeat=max_attractors),
            finder_configs.as_polars(repeat=max_attractors),
            df_init_conditions,
            solutions.as_polars(state_vector_labels=solution_state_labels),
        ],
        how="horizontal",
    )

    balanced = []
    unique_subharmonics = raw_data["detected_subharmonic"].unique()
    max_detected = unique_subharmonics.max()
    for sh in unique_subharmonics:
        group = raw_data.filter(pl.col("detected_subharmonic") == sh)
        limit = sh if sh != 0 else max_detected
        balanced.append(group.group_by("sim_label").head(limit))

    return pl.concat(balanced, how="vertical").sort(["sim_label", "attractor_label"])


def cluster_points(points, weights, distance_threshold=0.01, method="dbscan"):
    """
    Cluster points in a weighted feature space and return cluster centroids.

    The input points are scaled elementwise by ``weights`` before clustering,
    but centroids are computed in the original (unweighted) point space.

    Args:
        points (array-like): Array of shape ``(n_points, n_dims)`` containing the
            points to cluster.
        weights (array-like): Per-dimension scaling factors broadcastable to
            ``points`` (typically shape ``(n_dims,)``).
        distance_threshold (float, optional): Clustering scale parameter.
            - For ``method="dbscan"`` / ``"dbscan_cuml"``: used as DBSCAN ``eps``.
            - For ``method="agglomerative-clustering"``: used as
              ``AgglomerativeClustering(distance_threshold=...)`` with
              ``n_clusters=None`` and ``linkage="ward"``.
        method (str, optional): Clustering backend. One of
            ``{"dbscan", "dbscan_cuml", "agglomerative-clustering"}``.

    Returns:
        tuple:
            - nclusters (int): Number of unique cluster labels.
            - labels (np.ndarray): Array of shape ``(n_points,)`` with the cluster
              label for each point.
            - centroids (np.ndarray): Array of shape ``(nclusters, n_dims)`` with
              the mean of points belonging to each cluster label.

    Notes:
        - DBSCAN is configured with ``min_samples=1``, so points are not treated
          as noise (i.e. labels are expected to be ``0..nclusters-1``).
        - ``method="dbscan_cuml"`` requires ``cuml`` to be installed and importable.
    """
    if len(points) > 1:
        X = points * weights  # Array CPU
        if method == "agglomerative-clustering":
            db = AgglomerativeClustering(
                distance_threshold=distance_threshold, n_clusters=None, linkage="ward"
            ).fit(X)
        if method == "dbscan":
            db = DBSCAN(eps=distance_threshold, min_samples=1).fit(
                X
            )  # applique la methode fit() de l'objet DBSAN de scikit-learn; fit(X) effectue
            # le clustering DBSCAN a partir des caracteristiques ou de la matrice de distance
        if method == "dbscan_cuml":
            db = cuml.DBSCAN(eps=distance_threshold, min_samples=1).fit(
                X
            )  # CUDA version
        labels = (
            db.labels_
        )  # etiquettes des clusters pour chaque point de l'ensemble de donnees donne a fit(X). etiquette -1 si bruite.
        nclusters = np.unique(labels).size
        centroids = []
        for c in range(nclusters):
            centroids.append(points[labels == c].mean(axis=0))
        centroids = np.array(centroids)
        return nclusters, labels, centroids

    else:
        centroids = points.copy()
        return 1, np.zeros(1, dtype=np.int32), centroids


def detect_orbits(
    problem_class: NamedTuple,
    simulations: pl.DataFrame,
    ode_params_labels: list,
    attractor_state_vec_labels: list,
    state_vec_labels: list,
    distance_threshold: float = 0.01,
    clustering_method: str = "dbscan",
):
    """Cluster attractor samples per configuration and map them to orbit IDs.

    Workflow
    1. Group rows by ODE parameters, detected subharmonic, and target frequency
       so each configuration is processed independently.
    2. Cluster the recorded attractor points in the weighted state space to
       estimate unique attractors for the group.
    3. Assign globally unique attractor labels and record which simulations
       share the same attractor tuple (one orbit).
    4. Return tidy Polars DataFrames for attractors and the simulation->orbit map.
    """

    def canonicalize_cluster_sequence(sequence: np.ndarray) -> tuple[int, ...]:
        """Rotate a sequence so the smallest label appears first (orbit invariant)."""
        sequence = np.asarray(sequence, dtype=int)
        if sequence.size == 0:
            return tuple()
        rotation = int(np.argmin(sequence))
        return tuple(np.roll(sequence, -rotation))

    next_attractor_label = 0
    group_labels = ode_params_labels + ["detected_subharmonic", "target_frequency"]
    attractor_columns = (
        state_vec_labels
        + ode_params_labels
        + [
            "detected_subharmonic",
            "attractor_label",
            "orbit_label",
            "target_frequency",
        ]
    )
    attractors = {col: [] for col in attractor_columns}
    sim_to_attractor_tuple = {}
    attractor_tuple_to_sims = defaultdict(list)

    for keys, group in simulations.group_by(group_labels):
        params = dict(zip(group_labels, keys))
        sh = params["detected_subharmonic"]
        if sh <= 0 or group.height == 0:
            continue

        # Process simulations sharing the same ODE parameters/subharmonics.
        group = group.sort(["sim_label", "attractor_label"])
        points = group.select(attractor_state_vec_labels).to_numpy()
        problem = problem_class(**{k: params[k] for k in ode_params_labels})
        weights = problem.state_weights()
        nclusters, labels, centroids = cluster_points(
            points,
            weights,
            distance_threshold=distance_threshold,
            method=clustering_method,
        )

        # Offset cluster labels so they stay unique across groups.
        labels = labels + next_attractor_label
        attractor_labels = np.arange(nclusters, dtype=int) + next_attractor_label

        for label, centroid in zip(attractor_labels, centroids):
            for k in ode_params_labels:
                attractors[k].append(params[k])
            for value, state_label in zip(centroid, state_vec_labels):
                attractors[state_label].append(value)
            attractors["detected_subharmonic"].append(sh)
            attractors["attractor_label"].append(int(label))
            attractors["target_frequency"].append(params["target_frequency"])

        next_attractor_label += nclusters

        # Build the attractor sequence followed by each simulation to define orbits.
        sim_labels = group["sim_label"][::sh].to_numpy()
        label_sequences = labels.reshape(-1, sh)
        for sim_label, sequence in zip(sim_labels, label_sequences):
            canonical = canonicalize_cluster_sequence(sequence)
            sim_to_attractor_tuple[int(sim_label)] = canonical
            attractor_tuple_to_sims[canonical].append(int(sim_label))

    # Every unique attractor tuple corresponds to one orbit ID.
    orbit_attractor_map = {
        orbit_id: attractor_tuple
        for orbit_id, attractor_tuple in enumerate(attractor_tuple_to_sims.keys())
    }
    orbit_id_lookup = {
        tuple_: orbit_id for orbit_id, tuple_ in orbit_attractor_map.items()
    }
    attractor_to_orbit = {
        attractor_label: orbit_id
        for orbit_id, attractor_tuple in orbit_attractor_map.items()
        for attractor_label in attractor_tuple
    }
    attractors["orbit_label"].extend(
        attractor_to_orbit[aid] for aid in attractors["attractor_label"]
    )

    sim_orbit_df = pl.DataFrame(
        {
            "sim_label": list(sim_to_attractor_tuple.keys()),
            "orbit_label": [
                orbit_id_lookup[tuple_] for tuple_ in sim_to_attractor_tuple.values()
            ],
        }
    )
    attractors_df = pl.DataFrame(attractors)
    return attractors_df, sim_orbit_df
