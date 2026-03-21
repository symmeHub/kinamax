# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 3: Reconstruct One-Period Orbits From The Attractors
#
# This notebook starts from the orbit representatives produced in Step 2 and
# integrates one drive period from each attractor state.
#
# It writes:
#
# - one parquet file per reconstructed orbit in `outputs/orbits_from_attractors/`,
# - an updated `outputs/orbits.parquet`,
# - a compact per-orbit summary in `outputs/orbit_data.parquet`.

# %%
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from models import H46_EM_Problem

problem_class = H46_EM_Problem


# %% [markdown]
# ## Helper To Reorder Simulations By Orbit
#
# Each detected attractor belongs to one orbit label. This helper reconstructs
# the matching simulation metadata in the same order as the attractor table so
# that parameters and initial times remain aligned.

# %%
def stack_simulations(orbits, sim_orbit, simulations):
    attractor_orbit_map = dict(
        zip(orbits["attractor_label"].to_list(), orbits["orbit_label"].to_list())
    )
    orbit_to_sim = sim_orbit.group_by("orbit_label").first()
    orbit_sim_map = dict(
        zip(orbit_to_sim["orbit_label"].to_list(), orbit_to_sim["sim_label"].to_list())
    )
    attractor_sim_map = {k: orbit_sim_map[v] for k, v in attractor_orbit_map.items()}

    order_df = pl.DataFrame(
        {"sim_label": list(attractor_sim_map.values())}
    ).with_row_index("order")

    unique_sims = simulations.unique(subset=["sim_label"], keep="first")

    stacked = (
        order_df.join(unique_sims, on="sim_label", how="left")
        .sort("order")
        .drop("order")
    )
    return stacked


# %% [markdown]
# ## One-Period Orbit Integrator
#
# For each attractor state, the calculator integrates one forcing period and
# returns `samples_per_period` snapshots.

# %%
class OrbitCalculator(NamedTuple):
    samples_per_period: int = np.array(60)

    def calculate_orbit(
        self, problem, Xa, init_time, target_frequency, init_time_step=1e-4
    ):
        """
        Calculate orbits from attractors.
        Args:
            Xa (jnp.ndarray): Attractor states.
            init_time (jnp.ndarray): Initial times.
            problem (H46Problem): Problem instance.
        Returns:
            jnp.ndarray: Orbits.
        """
        Ns = self.samples_per_period
        solver = Tsit5()
        controller = PIDController(rtol=1e-8, atol=1e-9)
        term = ODETerm(problem.rhs)
        t0 = init_time
        fd = target_frequency
        Td = 1.0 / fd
        t1 = t0 + Td
        dt0 = init_time_step
        # print(f"Calculating orbit from t={t0} to t={t1} with dt0={dt0} and N={Ns}" )
        tg = jnp.arange(0, np.array(Ns))
        # tg[0] = 0.
        # tg[1] = 1.0
        ts = t0 + tg * (Td / Ns)
        #ts = ts.at[-1].set(t1)
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0=Xa,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )
        return sol.ys


# %% [markdown]
# ## Load The Outputs Of The Previous Stages
#
# This stage consumes the simulation table from Step 1 and the orbit metadata
# from Step 2.

# %%
example_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
working_dir = example_dir / "outputs"

simulations = pl.read_parquet(working_dir / "simulations.parquet")
orbits = pl.read_parquet(working_dir / "orbits.parquet")
sim_orbit = pl.read_parquet(working_dir / "sim_orbit.parquet")
state_vec_labels = problem_class.state_vector_labels
attractor_state_vec_labels = [f"{k}a" for k in state_vec_labels]
ode_params_labels = problem_class.params_labels
stacked_sims = stack_simulations(orbits, sim_orbit, simulations)
ode_params = {k: jnp.array(stacked_sims[k].to_numpy()) for k in ode_params_labels}
target_frequencies = jnp.array(stacked_sims["target_frequency"].to_numpy())
init_times = jnp.array(stacked_sims["init_time"].to_numpy())
init_time_steps = jnp.array(stacked_sims["init_time_step"].to_numpy())
Xa = jnp.array(orbits[state_vec_labels].to_numpy())

problems = problem_class(**ode_params)


# %% [markdown]
# ## Integrate One Period From Every Attractor
#
# `vmap` applies the one-period integrator to every detected attractor while
# keeping the corresponding problem parameters aligned.

# %%
calculator = OrbitCalculator(samples_per_period=200)
calculator_fn = jax.jit(jax.vmap(calculator.calculate_orbit))
calculated_orbits = np.array(
    calculator_fn(
        problems,
        Xa=Xa,
        init_time=init_times,
        target_frequency=target_frequencies,
        init_time_step=init_time_steps,
    )
)


# %% [markdown]
# ## Write One File Per Orbit And Update The Summary Tables
#
# The detailed trajectories are stored separately, while the aggregate orbit
# data are folded back into the main parquet tables.

# %%
orbits_labels = np.asarray(orbits["orbit_label"])
attractors_labels = np.array(orbits["attractor_label"])
orbits_dir = working_dir / "orbits_from_attractors"
orbits_dir.mkdir(parents=True, exist_ok=True)

orbits_from_attractors = {}
for i in range(len(calculated_orbits)):
    orbit_label = orbits_labels[i]
    attractor_label = attractors_labels[i]
    orbit_data = calculated_orbits[i]
    orbit_df = pl.from_numpy(orbit_data, schema=state_vec_labels)
    orbits_from_attractors[(orbit_label, attractor_label)] = orbit_df
    orbit_df.write_parquet(
        orbits_dir / f"orbit_{orbit_label}_attractor_{attractor_label}.parquet"
    )

dicts = [
    {"orbit_label": k[0], "attractor_label": k[1], "Eh": v.item(-1, "Eh")}
    for (k, v) in orbits_from_attractors.items()
]

energy_per_period = pl.from_dicts(dicts)
orbits = orbits.drop("Eh").join(
    energy_per_period, on=["orbit_label", "attractor_label"], how="right"
)

orbits.write_parquet(working_dir / "orbits.parquet")


unique_orbits = orbits.unique(subset=["orbit_label"], keep="first")[
    "orbit_label",
    "fd",
    "detected_subharmonic",
]
orbit_energy = orbits["orbit_label", "Eh"].group_by("orbit_label").mean()
orbit_data = unique_orbits.join(orbit_energy, on="orbit_label", how="inner")
orbit_data = orbit_data.with_columns((pl.col("Eh") * pl.col("fd")).alias("Ph"))
orbit_data.write_parquet(working_dir / "orbit_data.parquet")

orbit_data
