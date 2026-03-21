"""Problem definitions expressed as data containers plus standalone operators."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["H46Problem"]


class H46Problem:
    """Namespace holding the H46 problem data layouts and static helpers."""

    state_vector_labels: tuple[str, ...] = ("x", "dotx", "Eh")
    params_labels: tuple[str, ...] = ("xw", "w0", "Ad", "Q", "fd")

    class Params(NamedTuple):
        """Physical parameters of the driven H46 oscillator."""

        xw: jax.Array = jnp.array(0.5e-3)
        fd: jax.Array = jnp.array(50.0)
        w0: jax.Array = jnp.array(121.0)
        Q: jax.Array = jnp.array(87.0)
        Ad: jax.Array = jnp.array(2.5)

    @staticmethod
    def state_weights(problem: "H46Problem.Params") -> jax.Array:
        """Return per-state scaling weights aligned with ``[x, dotx, Eh]``."""
        return jnp.array([1.0 / problem.xw, 1.0 / (problem.w0 * problem.xw), 0.0])

    @staticmethod
    def rhs(
        problem: "H46Problem.Params",
        t: ArrayLike,
        X: ArrayLike,
        args: Any = None,
    ) -> jax.Array:
        """Evaluate the H46 oscillator right-hand side."""
        del args
        x, dotx, _Eh = jnp.asarray(X)
        wd = 2.0 * jnp.pi * problem.fd
        ddotx = (
            -(jnp.pow(problem.w0, 2)) / 2.0 * (jnp.pow(x / problem.xw, 2) - 1.0) * x
            - problem.w0 / problem.Q * dotx
            + problem.Ad * jnp.sin(wd * jnp.asarray(t))
        )
        Ph = problem.w0 / problem.Q * dotx**2
        return jnp.array([dotx, ddotx, Ph])
