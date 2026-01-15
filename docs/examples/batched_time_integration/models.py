from dataclasses import dataclass
from typing import ClassVar
from kinamax.core import Container
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

@register_dataclass
@dataclass
class H46_EM_Problem(Container):
    """
    H46 Problem
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
    M :jax.Array = 17.3e-3
    state_vector_labels: ClassVar = ["x", "dotx", "v", "Ev", "Eh"]
    params_labels: ClassVar = ["xw", "w0", "Ad", "Q", "fd", "alpha", "C0", "R", "L", "M"]

    def state_weights(self):
        """
        Returns the state weights for the system.
        Returns:
            jnp.ndarray: State weights.
        """
        xw = self.xw
        w0 = self.w0
        return jnp.array([1.0 / xw, 1.0 / (w0 * xw), 1.0, 0.0, 0.0])

    def rhs(self, t, X, args=None):
        """
        Right-hand side of the ODE.
        Args:
            t (float): Time.
            X (jnp.ndarray): State vector.
            args (tuple, optional): Additional arguments. Defaults to None.
        Returns:
            jnp.ndarray: Derivative of the state vector.
        """
        xw = self.xw
        w0 = self.w0
        Q = self.Q
        fd = self.fd
        Ad = self.Ad
        alpha = self.alpha
        C0 = self.C0
        R = self.R
        L = self.L
        M = self.M
        x, dotx, v, Ev, Eh = X
        wd = 2.0 * jnp.pi * fd
        ddotx = (
            -(jnp.pow(w0, 2)) / 2.0 * (jnp.pow(x / xw, 2) - 1.0) * x
            - w0 / Q * dotx
            + Ad * jnp.sin(wd * t)
        )
        dotv = 2.0 * alpha / (L * C0) * x * dotx - v / (R * C0)
        Pv = dotx**2 * (w0 / Q) * M
        Ph = v**2 / R
        Xout = jnp.array([dotx, ddotx, dotv, Pv, Ph])
        return Xout