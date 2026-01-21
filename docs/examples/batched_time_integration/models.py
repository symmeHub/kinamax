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
    H46 benchmark problem with electromechanical (EM) coupling.

    The state is ordered as ``X = [x, dotx, v, Ev, Eh]``:
    - ``x``: mechanical displacement
    - ``dotx``: mechanical velocity
    - ``v``: electrical voltage
    - ``Ev``: cumulative viscous dissipation energy (integral of ``Pv``)
    - ``Eh``: cumulative Joule heating energy (integral of ``Ph``)

    Parameters (as used in :meth:`rhs`):
    - ``xw``: well position characteristic length
    - ``w0``: natural angular frequency
    - ``Q``: mechanical quality factor
    - ``fd``: drive frequency
    - ``Ad``: drive amplitude 
    - ``alpha``: EM coupling coefficient
    - ``C0``: capacitance
    - ``R``: electrical resistance
    - ``L``: beam length
    - ``M``: effective mass
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
    state_vector_labels: ClassVar = ["x", "dotx", "v", "Ev", "Eh"]
    params_labels: ClassVar = ["xw", "w0", "Ad", "Q", "fd", "alpha", "C0", "R", "L", "M"]

    def state_weights(self) -> jax.Array:
        """
        Return per-state scaling weights.

        The first two components scale displacement and velocity by the
        characteristic length ``xw`` and the frequency scale ``w0``.

        Returns:
            jax.Array: Array of shape ``(5,)`` aligned with ``[x, dotx, v, Ev, Eh]``.
        """
        xw = self.xw
        w0 = self.w0
        return jnp.array([1.0 / xw, 1.0 / (w0 * xw), 1.0, 0.0, 0.0])

    def rhs(self, t: jax.Array, X: jax.Array, args=None) -> jax.Array:
        """
        Right-hand side of the coupled ODE system.

        Notes:
            - ``args`` is accepted for API compatibility but is not used.
            - The last two states ``Ev`` and ``Eh`` integrate the instantaneous
              powers ``Pv`` and ``Ph`` respectively (i.e. ``dEv/dt = Pv``,
              ``dEh/dt = Ph``).

        Args:
            t: Time.
            X: State vector ``[x, dotx, v, Ev, Eh]``.
            args: Unused extra arguments.

        Returns:
            jax.Array: Time derivative ``dX/dt`` with the same ordering as ``X``.
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
