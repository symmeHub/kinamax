import numpy as np
import jax.numpy as jnp

from kinamax.integration.models import H46Problem


def test_h46_params_build_namedtuple_data():
    problem = H46Problem.Params(fd=jnp.array([10.0, 20.0]), Ad=jnp.array(3.0))

    assert isinstance(problem, H46Problem.Params)
    np.testing.assert_array_equal(np.asarray(problem.fd), np.array([10.0, 20.0]))
    np.testing.assert_allclose(np.asarray(problem.Ad), 3.0)


def test_h46_labels_and_weights_are_exposed_via_namespace():
    problem = H46Problem.Params(xw=jnp.array(0.5e-3), w0=jnp.array(100.0))

    assert H46Problem.state_vector_labels == ("x", "dotx", "Eh")
    assert H46Problem.params_labels == ("xw", "w0", "Ad", "Q", "fd")
    np.testing.assert_allclose(
        np.asarray(H46Problem.state_weights(problem)),
        np.array([2000.0, 20.0, 0.0]),
    )


def test_h46_rhs_matches_expected_equations():
    problem = H46Problem.Params(
        xw=jnp.array(2.0),
        fd=jnp.array(0.0),
        w0=jnp.array(4.0),
        Q=jnp.array(2.0),
        Ad=jnp.array(0.0),
    )
    state = np.array([1.0, 3.0, 7.0])

    rhs = H46Problem.rhs(problem, t=0.25, X=state)

    np.testing.assert_allclose(
        np.asarray(rhs),
        np.array([3.0, 0.0, 18.0]),
    )
