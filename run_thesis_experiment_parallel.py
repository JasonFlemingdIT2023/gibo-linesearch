"""
Parallelized experiment runner because of limited time constraint. More efficient run allocation.

Identical logic to run_thesis_experiment.py, but runsparallel using multiprocessing.Pool.

Usage:
    python run_thesis_experiment_parallel.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data \\
        --n_workers 24
"""

import os
import pickle
import argparse
from multiprocessing import Pool, cpu_count

import torch
import yaml

from src.loop import call_counter
from src.optimizers import BayesianGradientAscent
from src.model import DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_custom_bo
from src.synthetic_functions import generate_objective_from_gp_post
from run_thesis_experiment import build_optimizer, run_single


def worker_init():
    """Called once per worker process at Pool start.
       Note: Limits PyTorchs internal thread count to 1 so that n_workers processes
       each use 1 core, instead of n_workers * n_cpu_threads competing.
    """
    torch.set_num_threads(1)


def run_job(args):
    """Top level function required for pickle ability in multiprocessing.

    Unpacks the job tuple, skips if output already exists and runs run_single.
    """
    cfg, dim, run_index, seed, train_x, train_y, lengthscale, f_max, out_path = args

    if os.path.exists(out_path):
        print(f" dim={dim} run {run_index:03d} - exists, skipping.", flush=True)
        return

    print(f" dim={dim} run {run_index:03d} (seed={seed}) starting...", flush=True)
    try:
        result = run_single(
            cfg=cfg,
            dim=dim,
            seed=seed,
            train_x=train_x,
            train_y=train_y,
            lengthscale=lengthscale,
            f_max=f_max,
        )
        with open(out_path, "wb") as fh:
            pickle.dump(result, fh)
        regret_final = result["regret_per_eval"][-1] if result["regret_per_eval"] else float("nan")
        n_iters = len(result["f_values"])
        print(
            f" dim={dim} run {run_index:03d} done. "
            f"iters={n_iters}, regret={regret_final:.4f}",
            flush=True,
        )
    except Exception as e:
        print(f" dim={dim} run {run_index:03d} ERROR: {e}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Parallelized thesis experiment runner.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to pre-generated data directory.")
    parser.add_argument("--n_workers", type=int, default=None,help="Number of parallel workers (default are all CPUs).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Workers receive tensors
    data_dir = args.data_dir
    train_x_dict = torch.load(os.path.join(data_dir, "train_x.pt"))
    train_y_dict = torch.load(os.path.join(data_dir, "train_y.pt"))
    lengthscales_dict = torch.load(os.path.join(data_dir, "lengthscales.pt"))
    f_max_dict = torch.load(os.path.join(data_dir, "f_max.pt"))

    dimensions = cfg["dimensions"]
    n_runs = cfg["n_runs"]
    seed_start = cfg.get("seed_start", 0)
    results_dir = cfg["results_dir"]
    name = cfg["name"]

    # Build job list -> one entry per (dim, run).
    jobs = []
    for dim in dimensions:
        dim_dir = os.path.join(results_dir, name, f"dim_{dim}")
        os.makedirs(dim_dir, exist_ok=True)
        for run_index in range(n_runs):
            seed = seed_start + run_index
            out_path = os.path.join(dim_dir, f"run_{run_index:03d}.pkl")
            jobs.append((
                cfg,
                dim,
                run_index,
                seed,
                train_x_dict[dim][run_index],
                train_y_dict[dim][run_index],
                lengthscales_dict[dim],
                float(f_max_dict[dim][run_index]),
                out_path,
            ))

    # Skips already finished jobs before launching the workers.
    pending = [j for j in jobs if not os.path.exists(j[-1])]

    n_workers = min(args.n_workers or cpu_count(), len(pending)) if pending else 1

    print(f"\nExperiment: {name}")
    print(f"Mode: {cfg['inner_loop_mode']}")
    print(f"Dimensions: {dimensions}")
    print(f"Total jobs: {len(jobs)}  ({len(jobs) - len(pending)} already done, {len(pending)} pending)")
    print(f"Workers: {n_workers}\n")

    if not pending:
        print("All jobs already completed.")
        return

    with Pool(processes=n_workers, initializer=worker_init) as pool:
        pool.map(run_job, pending)

    print(f"\nDone. Results: {os.path.join(results_dir, name)}")


if __name__ == "__main__":
    main()

