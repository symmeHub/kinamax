import jax
import numpy as np
import polars as pl
from kinamax.core import H46Problem, cluster_points
import os

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)  # DOUBLE PRECISION




simulations = pl.read_parquet("data/simulations.parquet")
if "attractor_id" in simulations.columns:
    simulations = simulations.drop(
        "attractor_id"
    )  # Remove old attractor_id if it exists


state_vec_labels = H46Problem.state_vector_labels
ode_params_labels = H46Problem.params_labels

attractor_id = 0
group_labels = ode_params_labels + ["subharmonic", "target_frequency"]
attractors = {
    k: []
    for k in state_vec_labels
    + ode_params_labels
    + ["subharmonic", "attractor_id", "target_frequency"]
}
solutions_attractors = {"sim_id": [], "attractor_id": []}
for keys, group in simulations.group_by(group_labels):
    params = dict(zip(group_labels, keys))
    ode_params = {k: params[k] for k in ode_params_labels}
    fd = params["fd"]
    # print(f"Processing {params} with {len(group)} points")
    sh = params["subharmonic"]
    gsid = group["sim_id"]
    if sh > 0:
        points = group.select(state_vec_labels).to_numpy()
        problem = H46Problem(**ode_params)
        weights = problem.state_weights()
        nclusters, labels, centroids = cluster_points(
            points, weights, distance_threshold=0.01, method="dbscan"
        )
        for i in range(nclusters):
            for k in ode_params_labels:
                attractors[k].append(params[k])
            for j, k in enumerate(state_vec_labels):
                attractors[k].append(centroids[i, j])
            attractors["subharmonic"].append(sh)
            loc = np.where(labels == i)[0]
            solutions_attractors["attractor_id"].append(
                [attractor_id] * np.ones(len(loc), dtype=np.int32)
            )
            solutions_attractors["sim_id"].append(gsid[loc])
            attractors["attractor_id"].append(attractor_id)
            attractors["target_frequency"].append(params["target_frequency"])
            attractor_id += 1
    else:
        solutions_attractors["sim_id"].append(gsid)
        solutions_attractors["attractor_id"].append(np.zeros(len(gsid), dtype=np.int32))

attractors = pl.DataFrame(attractors)
solutions_attractors["sim_id"] = np.concatenate(solutions_attractors["sim_id"])
solutions_attractors["attractor_id"] = np.concatenate(
    solutions_attractors["attractor_id"]
)
solutions_attractors = pl.DataFrame(solutions_attractors).sort("sim_id")
simulations = simulations.join(solutions_attractors, on="sim_id", maintain_order="left")


attractors.write_parquet("data/attractors.parquet")
simulations.write_parquet("data/simulations.parquet")
