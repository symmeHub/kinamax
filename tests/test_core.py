from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from jax.tree_util import register_dataclass

from kinamax.core import (
    AttractorFinder,
    AttractorFinderConfig,
    AttractorFinderSolution,
    Container,
    cluster_points,
    convert_subharmonics_flags,
    detect_orbits,
    post_process_attractor_finder_results,
)


@dataclass
class DummyContainer(Container):
    a: np.ndarray
    b: np.ndarray


@register_dataclass
@dataclass
class OrbitTestProblem(Container):
    gain: jax.Array = field(default_factory=lambda: jnp.array(1.0))

    state_vector_labels: ClassVar[list[str]] = ["x0", "x1"]

    def rhs(self, t, X, args=None):
        return jnp.zeros_like(X)

    def state_weights(self):
        return jnp.ones(2)


def make_solution() -> AttractorFinderSolution:
    return AttractorFinderSolution(
        attractors=np.array(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ]
        ),
        detected_subharmonic=np.array([[1, 1], [2, 2]]),
        subharmonic_residual=np.array([[0.1, 0.1], [0.2, 0.2]]),
        minimum_residual=np.array([[0.1, 0.1], [0.2, 0.2]]),
        simulated_periods=np.array([[2, 2], [4, 4]]),
        simulated_time=np.array([[1.0, 1.0], [2.0, 2.0]]),
        final_flag=np.array([[1, 1], [1, 1]]),
        simulated_iterations=np.array([[1, 1], [2, 2]]),
        converged=np.array([[True, True], [True, True]]),
    )


def test_container_exports_numpy_and_polars():
    container = DummyContainer(
        a=np.array([[1.0, 2.0]]),
        b=np.array([[3.0], [4.0]]),
    )

    as_dict = container.as_dict()
    frame = container.as_polars(repeat=2)

    np.testing.assert_array_equal(as_dict["a"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(as_dict["b"], np.array([3.0, 4.0]))
    assert frame.shape == (4, 2)
    np.testing.assert_array_equal(frame["a"].to_numpy(), np.array([1.0, 1.0, 2.0, 2.0]))


def test_convert_subharmonics_flags_selects_detected_orders():
    flags = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    final_flags = np.array([1, 0, 1])
    targets = np.array([[1, 2, 3], [1, 2, 3], [2, 4, 6]])

    detected = convert_subharmonics_flags(flags, final_flags, targets)

    np.testing.assert_array_equal(detected, np.array([2, 0, 6], dtype=np.int32))


def test_attractor_finder_solution_flattens_consistently():
    solution = make_solution()

    as_dict = solution.as_dict(state_vector_labels=["ax", "ay"])
    as_polars = solution.as_polars(state_vector_labels=["ax", "ay"])

    np.testing.assert_array_equal(as_dict["sim_label"], np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(as_dict["attractor_label"], np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(as_dict["ax"], np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(as_dict["ay"], np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(solution.get_subharmonics(), np.array([1, 1, 2, 2]))
    assert as_polars.shape == (4, 12)


def test_attractor_finder_sizes_and_residual_helpers():
    finder = AttractorFinder(
        residuals_per_period=3,
        targetted_subharmonics=np.array([1, 2], dtype=int),
        max_periods=12,
    )
    trajectory = jnp.zeros((7, 2))
    weights = jnp.array([1.0, 2.0])

    residual = AttractorFinder.calculate_subharmonic_residual(
        subharmonic=1,
        X=trajectory,
        residuals_per_period=3,
        state_weights=weights,
    )

    assert finder.get_max_subharmonic() == 2
    assert finder.get_time_steps_number() == 13
    assert finder.get_max_shooting_iterations() == 3
    assert float(residual) == pytest.approx(0.0)


def test_find_attractors_converges_for_constant_problem():
    finder = AttractorFinder(
        residuals_per_period=2,
        targetted_subharmonics=np.array([1, 2], dtype=int),
        max_periods=8,
    )
    config = AttractorFinderConfig(
        init_time=jnp.array(0.0),
        init_time_step=jnp.array(1.0e-2),
        convergence_tol=jnp.array(1.0e-12),
        target_frequency=1.0,
        subharmonic_factor=10.0,
    )
    init_conditions = jnp.array([1.5, -2.0])

    _, _, _, solution = finder.find_attractors(OrbitTestProblem(), init_conditions, config)

    np.testing.assert_array_equal(np.asarray(solution.converged), np.array([True, True]))
    np.testing.assert_array_equal(np.asarray(solution.detected_subharmonic), np.array([1, 1]))
    np.testing.assert_allclose(np.asarray(solution.subharmonic_residual), 0.0)
    np.testing.assert_allclose(np.asarray(solution.attractors), np.array([[1.5, -2.0], [1.5, -2.0]]))


def test_post_process_balances_rows_by_detected_subharmonic():
    problems = OrbitTestProblem(gain=np.array([1.0, 2.0]))
    finder_configs = AttractorFinderConfig(
        init_time=np.array([0.0, 0.0]),
        init_time_step=np.array([1.0e-3, 1.0e-3]),
        convergence_tol=np.array([1.0e-6, 1.0e-6]),
        target_frequency=np.array([10.0, 10.0]),
        subharmonic_factor=np.array([10.0, 10.0]),
    )
    init_conditions = np.array([[0.0, 0.0], [1.0, 1.0]])
    solution = make_solution()

    processed = post_process_attractor_finder_results(
        problem_class=OrbitTestProblem,
        problems=problems,
        finder_configs=finder_configs,
        init_conditions=init_conditions,
        solutions=solution,
        target_subharmonics=np.array([1, 2]),
        solution_state_labels=["ax", "ay"],
    )

    assert processed.shape == (3, 20)
    np.testing.assert_array_equal(processed["sim_label"].to_numpy(), np.array([0, 1, 1]))
    np.testing.assert_array_equal(
        processed["detected_subharmonic"].to_numpy(),
        np.array([1, 2, 2]),
    )


def test_cluster_points_returns_centroids_and_rejects_unknown_methods():
    points = np.array([[0.0, 0.0], [0.005, 0.0], [1.0, 0.0]])
    weights = np.ones(2)

    nclusters, labels, centroids = cluster_points(points, weights, distance_threshold=0.01)

    assert nclusters == 2
    np.testing.assert_array_equal(labels, np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(centroids, np.array([[0.0025, 0.0], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        cluster_points(points, weights, method="unsupported")


def test_detect_orbits_groups_rotated_sequences_into_one_orbit():
    simulations = pl.DataFrame(
        {
            "gain": [1.0, 1.0, 1.0, 1.0, 2.0],
            "detected_subharmonic": [2, 2, 2, 2, 1],
            "target_frequency": [10.0, 10.0, 10.0, 10.0, 10.0],
            "sim_label": [0, 0, 1, 1, 2],
            "attractor_label": [0, 1, 0, 1, 0],
            "x0": [0.0, 1.0, 1.0, 0.0, 10.0],
            "x1": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    attractors, sim_orbit = detect_orbits(
        problem_class=OrbitTestProblem,
        simulations=simulations,
        ode_params_labels=["gain"],
        attractor_state_vec_labels=["x0", "x1"],
        state_vec_labels=["x0", "x1"],
        distance_threshold=0.01,
        clustering_method="dbscan",
    )

    orbit_labels = dict(zip(sim_orbit["sim_label"].to_list(), sim_orbit["orbit_label"].to_list()))
    assert attractors.shape == (3, 7)
    assert orbit_labels[0] == orbit_labels[1]
    assert orbit_labels[2] != orbit_labels[0]
