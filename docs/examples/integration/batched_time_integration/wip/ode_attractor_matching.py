import jax
import jax.numpy as jnp
#import cuml
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy as np
import polars as pl
import os
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
from jax import vmap, jit
from kinamax.core import Container, cluster_points, AttractorSubharmonicMatcher,  H46Problem
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
from typing import NamedTuple
import networkx as nx
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)  # DOUBLE PRECISION

attractors = pl.read_parquet("data/attractors.parquet")
simulations = pl.read_parquet("data/simulations.parquet")

state_vec_labels = H46Problem.state_vector_labels
ode_params_labels = H46Problem.params_labels
group_labels = ode_params_labels + ["subharmonic", "target_frequency"]
orbits_data = []
orbit_id_counter = 0
for keys, group in attractors.group_by(group_labels):
    params = dict(zip(group_labels, keys))
    target_fd = params["target_frequency"]
    target_sh = params["subharmonic"]
    target_period = 1.0 / target_fd
    attr = group.row(0, named=True)
    sim_id = simulations.filter(pl.col("attractor_id") == attr["attractor_id"]).row(
        0, named=True
    )["sim_id"]
    sim_config = simulations.row(sim_id, named=True)
    solver = Tsit5()
    controller = PIDController(rtol=1e-7, atol=1e-9)
    problem_params = {
        k: attr[k] for k in ode_params_labels if k in attr
    }  # Extract only the relevant parameters
    problem = H46Problem(**problem_params)
    dt0 = sim_config["init_time_step"]
    t0 = sim_config["init_time"]
    X0s = jnp.array(group.select(["Xa_0", "Xa_1"]).to_numpy())
    matcher = AttractorSubharmonicMatcher(
        problem=problem, solver=solver, controller=controller
    )
    orbits = matcher.calc_orbits_attractors(
        attractors=X0s, t0=t0, target_period=target_period, dt0=dt0, target_sh=target_sh
    )
    orbits_ids = []
    orbit_attractor_ids = []
    for orbit in orbits:
        orbit_id = jnp.ones(len(orbit), dtype=int) + orbit_id_counter
        orbit_id_counter += 1
        orbit_attractor_id = jnp.arange(len(orbit))
        orbits_ids.append(orbit_id)
        orbit_attractor_ids.append(orbit_attractor_id)
    orbits_ids = jnp.concatenate(orbits_ids, axis=0)
    orbit_attractor_ids = jnp.concatenate(orbit_attractor_ids, axis=0)
    attractor_count = len(orbits_ids)
    orbit_dic = {"orbit_id": orbits_ids, "orbit_attractor_id": orbit_attractor_ids}
    for k in group_labels:
        orbit_dic[k] = jnp.repeat(params[k], attractor_count)
    orbits2 = jnp.concatenate(orbits, axis=0)
    for i, k in enumerate(state_vec_labels):
        orbit_dic[k] = np.array(orbits2[:, i])
    orbit_dic["target_frequency"] = jnp.repeat(target_fd, attractor_count)
    orbit_dic["subharmonic"] = jnp.repeat(target_sh, attractor_count)
    orbit_dic_np = {k: np.array(v) for k, v in orbit_dic.items()}
    orbit_df = pl.DataFrame(orbit_dic_np)
    orbits_data.append(orbit_df)

orbits_df = pl.concat(orbits_data, how="vertical").sort("orbit_id")
orbits_df.write_parquet("data/orbits.parquet")
