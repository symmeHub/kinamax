"""Time-integration problem definitions and reference models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike

from .core import Container

__all__ = ["H46Problem", "H46_EM_Problem"]


class H46Problem:
    """Namespace holding the H46 problem data layouts and static helpers."""

    state_vector_labels: tuple[str, ...] = ("x", "dotx", "Eh")
    params_labels: tuple[str, ...] = ("xw", "w0", "Ad", "Q", "fd")

    class Params(NamedTuple):
        """Physical parameters of the driven H46 oscillator.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> problem = H46Problem.Params(fd=jnp.array(50.0), Ad=jnp.array(2.5))
        >>> problem.fd
        Array(50., dtype=float32, weak_type=True)
        """

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


@register_dataclass
@dataclass
class H46_EM_Problem(Container):
    """H46 benchmark problem with electromechanical coupling.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> problem = H46_EM_Problem(fd=jnp.linspace(20.0, 50.0, 11), Ad=2.5)
    >>> problem.fd.shape
    (11,)
    """

    xw: jax.Array = 0.5e-3
    fd: jax.Array = 50.0
    w0: jax.Array = 121.0
    Q: jax.Array = 87.0
    Ad: jax.Array = 2.5
    alpha: jax.Array = 0.068
    C0: jax.Array = 1.05e-6
    R: jax.Array = 7.83e3
    L: jax.Array = 25e-3
    M: jax.Array = 17.3e-3

    state_vector_labels: ClassVar[list[str]] = ["x", "dotx", "v", "Ev", "Eh"]
    params_labels: ClassVar[list[str]] = [
        "xw",
        "w0",
        "Ad",
        "Q",
        "fd",
        "alpha",
        "C0",
        "R",
        "L",
        "M",
    ]

    def state_weights(self) -> jax.Array:
        """Return per-state scaling weights aligned with ``[x, dotx, v, Ev, Eh]``."""
        return jnp.array([1.0 / self.xw, 1.0 / (self.w0 * self.xw), 1.0, 0.0, 0.0])

    def rhs(self, t: jax.Array, X: jax.Array, args: Any = None) -> jax.Array:
        """Evaluate the electromechanically coupled H46 right-hand side."""
        del args
        x, dotx, v, _Ev, _Eh = X
        wd = 2.0 * jnp.pi * self.fd
        ddotx = (
            -(jnp.pow(self.w0, 2)) / 2.0 * (jnp.pow(x / self.xw, 2) - 1.0) * x
            - self.w0 / self.Q * dotx
            + self.Ad * jnp.sin(wd * t)
        )
        dotv = 2.0 * self.alpha / (self.L * self.C0) * x * dotx - v / (self.R * self.C0)
        Pv = dotx**2 * (self.w0 / self.Q) * self.M
        Ph = v**2 / self.R
        return jnp.array([dotx, ddotx, dotv, Pv, Ph])
