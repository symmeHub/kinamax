from dataclasses import dataclass
from typing import ClassVar
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from kinamax.core import Container

@register_dataclass
@dataclass
class H46Problem(Container):
    """
    H46 Problem
    """

    xw: jax.Array = 0.5e-3
    fd: jax.Array = 50.0
    w0: jax.Array = 121.0
    Q: jax.Array = 87.0
    Ad: jax.Array = 2.5

    state_vector_labels: ClassVar = ["x", "dotx", "Eh"]
    params_labels: ClassVar = ["xw", "w0", "Ad", "Q", "fd"]
    # label_col: ClassVar = "problem_id"

    def state_weights(self):
        """
        Returns the state weights for the system.
        Returns:
            jnp.ndarray: State weights.
        """
        xw = self.xw
        w0 = self.w0
        return jnp.array([1.0 / xw, 1.0 / (w0 * xw), 0.0])

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
        x, dotx, Eh = X
        wd = 2.0 * jnp.pi * fd
        ddotx = (
            -(jnp.pow(w0, 2)) / 2.0 * (jnp.pow(x / xw, 2) - 1.0) * x
            - w0 / Q * dotx
            + Ad * jnp.sin(wd * t)
        )
        Ph = w0 / Q * dotx**2
        Xout = jnp.array([dotx, ddotx, Ph])
        return Xout