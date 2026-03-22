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
# # Linear Oscillator: Amplitude And Phase From HBM
#
# This tutorial uses the simplest possible harmonic-balance problem: a linear
# damped oscillator driven by a single harmonic acceleration,
#
# $$
# \ddot{x} + \frac{w_0}{Q} \dot{x} + w_0^2 x = A \cos(\omega_d t).
# $$
#
# For this system the steady-state response is known analytically, so it is a
# good test case for the coefficient conventions used in `kinamax.hbm`.
#
# The goal is to:
#
# 1. define the residual Fourier series of the oscillator,
# 2. solve `R(X) = 0` for the harmonic coefficients,
# 3. recover amplitude and phase from the HBM coefficients,
# 4. compare the result with the classical transfer-function formula.

# %%
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
from jax import config
import jax.numpy as jnp
from IPython.display import HTML, display
import optimistix as optx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kinamax.hbm import (
    FourierCoeffs,
    coeffs_derivative,
    coeffs_to_complex,
    coeffs_to_time_signal,
)

config.update("jax_enable_x64", True)


# %% [markdown]
# ## Choose The Oscillator Parameters
#
# The mass-normalized model is fully described by the natural pulsation `w0`,
# the quality factor `Q`, and the forcing amplitude `A`, which is now an
# acceleration amplitude.

# %%
class LinearOscillatorParams(NamedTuple):
    w0: float
    Q: float
    A: float
    wd: jax.Array


frequency_ratio = jnp.linspace(0.2, 2.0, 400)
params = LinearOscillatorParams(
    w0=float(2.0 * jnp.pi),
    Q=10.0,
    A=1.0,
    wd=frequency_ratio * (2.0 * jnp.pi),
)


# %% [markdown]
# ## Define The Residual Fourier Series
#
# We seek a one-harmonic periodic response,
#
# $$
# x(t) = a_1 \cos(\omega_d t) + b_1 \sin(\omega_d t),
# $$
#
# the real coefficient vector is
#
# $$
# X = [a_0, a_1, b_1] = [0, a_1, b_1].
# $$
#
# Harmonic balance does not solve the transfer function directly. Instead it
# enforces cancellation of the residual Fourier series
#
# $$
# R(X) = \ddot{X} + \frac{w_0}{Q} \dot{X} + w_0^2 X - A,
# $$
#
# where the forcing coefficients are simply
#
# $$
# A = [0, A, 0].
# $$
#
# Because `kinamax.hbm.coeffs_derivative(...)` differentiates directly in the
# coefficient space, the residual can be built without leaving the Fourier
# representation. Here the drive pulsation sweep is stored in the `wd` leaf of
# a single parameter object, and `vmap` acts only on that leaf.

# %%
def balance_residual(X: jax.Array, params: LinearOscillatorParams) -> jax.Array:
    return (
        coeffs_derivative(X, params.wd, order=2)
        + (params.w0 / params.Q) * coeffs_derivative(X, params.wd, order=1)
        + params.w0**2 * X
        - jnp.array([0.0, params.A, 0.0])
    )

# %% [markdown]
# ## Solve The Harmonic-Balance Equations
#
# We now solve
#
# $$
# R(X) = 0
# $$
#
# with `optimistix.root_find(...)`.
#
# The nonlinear solver is not the point of this tutorial, so the root-finding
# details stay minimal. The important object is the Fourier residual `R(X)`.

# %%
solver = optx.Newton(rtol=1.0e-12, atol=1.0e-12)


def solve_coeffs(X0: jax.Array, params: LinearOscillatorParams) -> jax.Array:
    solution = optx.root_find(
        balance_residual,
        solver,
        y0=X0,
        args=params,
        max_steps=32,
    )
    return solution.value

X0 = jnp.zeros(3)
jsolve_coeffs = jax.jit(
    jax.vmap(
        solve_coeffs,
        in_axes=(None, LinearOscillatorParams(w0=None, Q=None, A=None, wd=0)),
    )
)
coeffs = jsolve_coeffs(X0, params)

initial_residual_norm = jnp.max(
    eqx.filter_vmap(lambda params: jnp.linalg.norm(balance_residual(jnp.zeros(3), params)))(
        params
    )
)
final_residual_norm = jnp.max(
    eqx.filter_vmap(lambda X, params: jnp.linalg.norm(balance_residual(X, params)))(
        coeffs, params
    )
)

print(f"Maximum residual norm before solve: {float(initial_residual_norm):.3e}")
print(f"Maximum residual norm after solve: {float(final_residual_norm):.3e}")


# %% [markdown]
# ## Convert Coefficients To Complex Phasors
#
# `kinamax.hbm.coeffs_to_complex(...)` converts the real coefficient vector to
# the positive-frequency complex phasor vector
#
# $$
# [C_0, C_1] = [a_0, a_1 - i b_1].
# $$
#
# For the first harmonic,
#
# $$
# C_1 = A e^{-i \phi},
# $$
#
# so:
#
# - the amplitude is `abs(C1)`,
# - the phase lag is `-angle(C1)`.

# %%
phasors = jax.vmap(coeffs_to_complex)(coeffs)
response_phasor = phasors[:, 1]

amplitude_hbm = jnp.abs(response_phasor)
phase_hbm = -jnp.angle(response_phasor)


# %% [markdown]
# ## Compare With The Classical Analytical Response
#
# The steady-state transfer function of the same oscillator is
#
# $$
# H(i \omega_d) = \frac{1}{w_0^2 - \omega_d^2 + i \frac{w_0}{Q} \omega_d},
# $$
#
# so the displacement phasor is `A * H(i omega_d)`.

# %%
theoretical_phasor = params.A / (
    params.w0**2
    - params.wd**2
    + 1j * (params.w0 / params.Q) * params.wd
)
amplitude_theory = jnp.abs(theoretical_phasor)
phase_theory = -jnp.angle(theoretical_phasor)

amplitude_error = jnp.max(jnp.abs(amplitude_hbm - amplitude_theory))
phase_error = jnp.max(jnp.abs(phase_hbm - phase_theory))

print(f"Maximum amplitude error: {float(amplitude_error):.3e}")
print(f"Maximum phase error: {float(phase_error):.3e} rad")


# %% [markdown]
# ## Check The Harmonic-Balance Residual
#
# The solved coefficients should now make the residual Fourier series almost
# vanish over the whole sweep.

# %%
residuals = eqx.filter_vmap(balance_residual)(coeffs, params)
residual_norm = jnp.max(jnp.linalg.norm(residuals, axis=1))
print(f"Maximum HBM residual norm over the sweep: {float(residual_norm):.3e}")


# %% [markdown]
# ## Plot Amplitude And Phase
#
# The two curves should be visually indistinguishable, since for this linear
# problem the one-harmonic balance model is exact.

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=("Amplitude", "Phase Lag"),
)

fig.add_trace(
    go.Scatter(
        x=frequency_ratio,
        y=amplitude_theory,
        name="Theory",
        mode="lines",
        line={"width": 3, "color": "#1f77b4"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=frequency_ratio,
        y=amplitude_hbm,
        name="HBM",
        mode="lines",
        line={"width": 2, "dash": "dash", "color": "#d62728"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=frequency_ratio,
        y=phase_theory,
        name="Theory",
        mode="lines",
        line={"width": 3, "color": "#1f77b4"},
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=frequency_ratio,
        y=phase_hbm,
        name="HBM",
        mode="lines",
        line={"width": 2, "dash": "dash", "color": "#d62728"},
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text="Frequency Ratio, omega_d / w0", row=2, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_yaxes(title_text="Phase Lag [rad]", row=2, col=1)
fig.update_layout(height=700, width=900, title="Linear Oscillator Response")
display(HTML(fig.to_html(include_plotlyjs="cdn")))


# %% [markdown]
# ## Reconstruct One Time-Domain Signal
#
# To close the loop, we reconstruct the time response near resonance and compare
# it with the applied acceleration input over one period.

# %%
resonance_index = int(jnp.argmin(jnp.abs(frequency_ratio - 1.0)))
wd_res = params.wd[resonance_index]
X_res = coeffs[resonance_index]
resonance_coeffs = FourierCoeffs(
    values=X_res,
    frequency=wd_res / (2.0 * jnp.pi),
)

response_signal = coeffs_to_time_signal(resonance_coeffs, oversample=64)
time = response_signal.time_grid
input_signal = params.A * jnp.cos(wd_res * time)

signal_fig = go.Figure()
signal_fig.add_trace(
    go.Scatter(
        x=time,
        y=input_signal,
        mode="lines",
        name="Acceleration Input",
        line={"width": 3, "color": "#444444"},
    )
)
signal_fig.add_trace(
    go.Scatter(
        x=time,
        y=response_signal.values,
        mode="lines",
        name="Response",
        line={"width": 3, "color": "#2ca02c"},
    )
)
signal_fig.update_layout(
    title="Time-Domain Reconstruction At Resonance",
    xaxis_title="Time [s]",
    yaxis_title="Signal Value",
    width=900,
    height=450,
)
display(HTML(signal_fig.to_html(include_plotlyjs=False)))
