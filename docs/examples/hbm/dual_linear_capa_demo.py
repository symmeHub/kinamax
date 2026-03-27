# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: science
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dual Linear Capacitive Oscillators With HBFFT
#
# This example updates a legacy HBFFT script to the current `kinamax.hbm` API.
# The model contains two linear oscillators, each coupled to a capacitive
# electrical branch. Harmonic balance is used directly on the steady-state ODE.
#
# The point of the example is to show how the modern `FourierCoeffs` container
# can be used inside the residual itself:
#
# - each unknown block is handled as a `FourierCoeffs`,
# - time derivatives are computed with `coeffs_derivative(...)`,
# - the residual is assembled with the linear algebra now defined on the
#   containers,
# - the solver still works on one flat JAX vector, because `optimistix`
#   naturally expects array states.
#
# We keep only one harmonic in the approach, so each periodic signal is described
# by the three real coefficients
#
# $$
# [a_0, a_1, b_1].
# $$

# %%
from __future__ import annotations

from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp
from IPython.display import HTML, display
import optimistix as optx
import plotly.graph_objects as go

from kinamax.hbm import FourierCoeffs, coeffs_derivative, coeffs_to_time_signal

config.update("jax_enable_x64", True)


# %% [markdown]
# ## Choose The Model Parameters
#
# The mechanical parameters are expressed with angular pulsations, while the HBM
# containers store the driving frequency in Hz. That keeps the user-facing
# frequency metadata easy to read, and the derivative helper converts it to
# angular frequency internally.


# %%
class DualLinearCapaParams(NamedTuple):
    M: float = 0.1
    gamma: float = 0.1
    w0: float = 100.0 * 2.0 * jnp.pi
    Q: float = 50.0
    r: float = 0.9
    c: float = 0.3
    Cp: float = 100.0e-9
    R1: float = 20.0e3
    R2: float = 10.0e3
    km2_1: float = 0.3
    km2_2: float = 0.3
    drive_frequency: float = 100.0


params = DualLinearCapaParams()
harmonic_order = 1
block_size = 2 * harmonic_order + 1


# %% [markdown]
# ## Split The Flat Unknown Vector Into HBM Blocks
#
# The unknown vector contains four periodic quantities:
#
# - the displacement of oscillator 1,
# - the displacement of oscillator 2,
# - the voltage of branch 1,
# - the voltage of branch 2.
#
# Each block is wrapped as a `FourierCoeffs` with the same driving frequency.


# %%
def split_state(
    X: jax.Array, params: DualLinearCapaParams
) -> tuple[FourierCoeffs, FourierCoeffs, FourierCoeffs, FourierCoeffs]:
    frequency = jnp.asarray(params.drive_frequency)
    x1 = FourierCoeffs(values=X[0:block_size], frequency=frequency)
    x2 = FourierCoeffs(values=X[block_size : 2 * block_size], frequency=frequency)
    v1 = FourierCoeffs(values=X[2 * block_size : 3 * block_size], frequency=frequency)
    v2 = FourierCoeffs(values=X[3 * block_size : 4 * block_size], frequency=frequency)
    return x1, x2, v1, v2


def stack_state(
    x1: FourierCoeffs,
    x2: FourierCoeffs,
    v1: FourierCoeffs,
    v2: FourierCoeffs,
) -> jax.Array:
    return jnp.concatenate([x1.values, x2.values, v1.values, v2.values])


# %% [markdown]
# ## Define The Harmonic-Balance Residual
#
# The model is linear, but the residual is written exactly as it would be for a
# more complicated system: differentiate the Fourier coefficients, assemble each
# equation in coefficient space, and then flatten the result back to one vector.
#
# The coupling coefficients are built from the electromechanical parameters:
#
# $$
# \alpha_1 = w_0 \sqrt{k_{m1}^2 M C_p}
# $$
#
# And
#
# $$
# \alpha_2 = r w_0 \sqrt{k_{m2}^2 M C_p}
# $$


# %%
def dual_linear_capa_residual(X: jax.Array, params: DualLinearCapaParams) -> jax.Array:
    tau1 = params.R1 * params.Cp
    tau2 = params.R2 * params.Cp
    alpha1 = jnp.sqrt(params.km2_1 * params.M * params.w0**2 * params.Cp)
    alpha2 = jnp.sqrt(params.km2_2 * params.M * (params.r * params.w0) ** 2 * params.Cp)

    x1, x2, v1, v2 = split_state(X, params)
    forcing = FourierCoeffs.cosine(
        amplitude=params.gamma,
        harmonic=1,
        order=harmonic_order,
        frequency=params.drive_frequency,
    )

    dx1 = coeffs_derivative(x1, order=1)
    dx2 = coeffs_derivative(x2, order=1)
    ddx1 = coeffs_derivative(x1, order=2)
    ddx2 = coeffs_derivative(x2, order=2)
    dv1 = coeffs_derivative(v1, order=1)
    dv2 = coeffs_derivative(v2, order=1)

    res1 = (
        ddx1
        + (params.w0 / params.Q) * dx1
        + params.w0**2 * x1
        + (alpha1 / params.M) * v1
        - forcing
    )
    res2 = (
        ddx2
        + (params.r * params.w0 / params.Q) * dx2
        + (params.r * params.w0) ** 2 * x2
        + (alpha2 / params.M) * v2
        - forcing
    )
    res3 = (alpha1 / params.Cp) * dx1 - dv1 - params.c * (dv1 - dv2) - v1 / tau1
    res4 = (alpha2 / params.Cp) * dx2 - dv2 - params.c * (dv2 - dv1) - v2 / tau2

    return stack_state(res1, res2, res3, res4)


# %% [markdown]
# ## Solve For The Steady-State Fourier Coefficients
#
# Because the system is linear, a Newton solve from a zero initial guess is
# enough. The example stays intentionally simple: one drive frequency, one
# solve, then direct post-processing of the Fourier coefficients.

# %%
solver = optx.Newton(rtol=1.0e-10, atol=1.0e-10)
X0 = jnp.zeros(4 * block_size)

solution = optx.root_find(
    dual_linear_capa_residual,
    solver,
    y0=X0,
    args=params,
    max_steps=32,
    throw=False,
)

success = bool(solution.result == optx.RESULTS.successful)
X1, X2, V1, V2 = split_state(solution.value, params)
residual_norm = float(
    jnp.linalg.norm(dual_linear_capa_residual(solution.value, params))
)

print(f"HBFFT success: {success}")
print(f"Drive frequency: {params.drive_frequency:.3f} Hz")
print(f"Residual norm: {residual_norm:.3e}")


# %% [markdown]
# ## Estimate The Electrical Power In Both Branches
#
# With the one-harmonic approach and zero DC voltage, the average dissipated power
# in each resistor is:
#
# $$
# P = \frac{a_1^2 + b_1^2}{2 R}.
# $$
#
# In our real coefficient layout, that is simply the squared Euclidean norm of
# `values[1:]`, divided by `2 R`.

# %%
power_1 = float(jnp.sum(V1.values[1:] ** 2) / (2.0 * params.R1))
power_2 = float(jnp.sum(V2.values[1:] ** 2) / (2.0 * params.R2))

print(f"Power branch 1: {power_1:.8e} W")
print(f"Power branch 2: {power_2:.8e} W")


# %% [markdown]
# ## Reconstruct One Period In The Time Domain
#
# To make the result easier to interpret, we reconstruct one period of the two
# mechanical responses and the two voltages. This also shows the intended
# back-and-forth workflow of HBFFT:
#
# - solve in frequency space,
# - reconstruct in time only when needed for interpretation or nonlinear terms.

# %%
x1_signal = coeffs_to_time_signal(X1, oversample=64)
x2_signal = coeffs_to_time_signal(X2, oversample=64)
v1_signal = coeffs_to_time_signal(V1, oversample=64)
v2_signal = coeffs_to_time_signal(V2, oversample=64)
time = x1_signal.time_grid

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=time,
        y=x1_signal.values,
        mode="lines",
        name="x1",
        line={"width": 3, "color": "#1f77b4"},
    )
)
fig.add_trace(
    go.Scatter(
        x=time,
        y=x2_signal.values,
        mode="lines",
        name="x2",
        line={"width": 3, "color": "#ff7f0e"},
    )
)
fig.add_trace(
    go.Scatter(
        x=time,
        y=v1_signal.values,
        mode="lines",
        name="v1",
        line={"width": 2, "dash": "dash", "color": "#2ca02c"},
    )
)
fig.add_trace(
    go.Scatter(
        x=time,
        y=v2_signal.values,
        mode="lines",
        name="v2",
        line={"width": 2, "dash": "dash", "color": "#d62728"},
    )
)
fig.update_layout(
    title="Dual Capacitive Oscillator Response Over One Period",
    xaxis_title="Time [s]",
    yaxis_title="Signal value",
    width=900,
    height=500,
)
display(HTML(fig.to_html(include_plotlyjs="cdn")))
