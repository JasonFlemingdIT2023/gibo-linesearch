"""
Experiment runner for thesis experiments.

Saves metrics per run in .pkl files.

Usage:
    python run_thesis_experiment.py \\
        -c experiments/thesis_experiments/configs/gibo_baseline.yaml \\
        -d experiments/thesis_experiments/data

Output:
    <results_dir>/<variant_name>/dim_<D>/run_<NNN>.pkl

Each .pkl contains a dict with keys:
    regret_per_eval, f_values, inner_loop_samples etc.
"""

import os
import pickle
import argparse

import torch
import yaml

from src.loop import call_counter
from src.optimizers import BayesianGradientAscent
from src.model import DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_custom_bo
from src.synthetic_functions import (generate_objective_from_gp_post, get_lengthscales, factor_hennig,)

# Helpers

def build_optimizer(cfg: dict, dim: int, params_init: torch.Tensor,
                    objective, lengthscale: torch.Tensor) -> BayesianGradientAscent:
    """Instantiate BayesianGradientAscent from thesis config dict."""
 
    # Resolve YAML placeholder strings to concrete values,,which matches the original GIBO config.evaluate() behaviour.
    n_max_raw = cfg["N_max"]
    n_max = 5 * dim if n_max_raw == "variable" else int(n_max_raw)

    max_samples_raw = cfg["max_samples_per_iteration"]
    if max_samples_raw == "dim_search_space":
        max_samples = dim
    else:
        max_samples = int(max_samples_raw)

    hypers = {
        "covar_module.base_kernel.lengthscale": lengthscale,
        "covar_module.outputscale": torch.tensor(cfg["gp_hypers"]["outputscale"]),
        "likelihood.noise": torch.tensor(cfg["noise_variance"]),
    }

    model_config = dict(
        N_max=n_max,
        ard_num_dims=dim,
        prior_mean=0.0,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
    )

    hyperparameter_config = dict(
        hypers=hypers,
        no_noise_optimization=cfg["no_noise_optimization"],
        optimize_hyperparameters=cfg["optimize_hyperparameters"],
    )

    optimizer_torch_config = {"lr": cfg["lr"]}
    lr_schedular = cfg.get("lr_schedular", None)

    # GradientAscent object instead of loop function
    return BayesianGradientAscent(
        params_init=params_init,
        objective=objective,
        max_samples_per_iteration=max_samples,
        OptimizerTorch=torch.optim.SGD,
        optimizer_torch_config=optimizer_torch_config,
        lr_schedular=lr_schedular,
        Model=DerivativeExactGPSEModel,
        model_config=model_config,
        hyperparameter_config=hyperparameter_config,
        optimize_acqf=optimize_acqf_custom_bo,
        optimize_acqf_config=dict(
            q=1,
            num_restarts=cfg["acqf_num_restarts"],
            raw_samples=cfg["acqf_raw_samples"],
        ),
        bounds=None,
        delta=cfg["delta"],
        epsilon_diff_acq_value=cfg.get("epsilon_diff_acq_value", None),
        generate_initial_data=None,
        normalize_gradient=cfg.get("normalize_gradient", False),
        standard_deviation_scaling=cfg.get("standard_deviation_scaling", False),
        verbose=cfg.get("verbose", False),
        inner_loop_mode=cfg["inner_loop_mode"],
        c1=cfg["c1"],
        c2=cfg["c2"],
        c_W=cfg["c_W"], 
        sigma_floor=cfg.get("sigma_floor", 0.1),
        alpha_min_factor=cfg.get("alpha_min_factor", 0.1),
        alpha_max_factor=cfg.get("alpha_max_factor", None),
        budget=cfg.get("budget", cfg.get("max_objective_calls", 300)),
    )

# Return a result dict via running one optimization object.
def run_single(cfg: dict, dim: int, seed: int,
               train_x: torch.Tensor, train_y: torch.Tensor,
               lengthscale: torch.Tensor, f_max: float) -> dict:

    torch.manual_seed(seed)

    objective_raw = generate_objective_from_gp_post(
        train_x=train_x,
        train_y=train_y,
        noise_variance=cfg["noise_variance"],
        gp_hypers={
            "covar_module.base_kernel.lengthscale": lengthscale,
            "covar_module.outputscale": torch.tensor(cfg["gp_hypers"]["outputscale"]),
        },
    )
    objective = call_counter(objective_raw)

    params_init = 0.5 * torch.ones(dim, dtype=torch.float32)
    optimizer = build_optimizer(cfg, dim, params_init, objective, lengthscale)
    mode = cfg["inner_loop_mode"]

    # Metric accumulators
    f_values = []
    inner_loop_samples = []  
    gradient_norms = []
    posterior_variance_trace = []
    calls_at_iteration = []
    n_steps_per_iter = []
    alpha_history_per_iter = []
    step_trigger_per_iter = []
    pwolfe_history_per_iter = []
    armijo_history_per_iter = []
    curvature_history_per_iter = []  

    max_calls = cfg["max_objective_calls"]

    while objective._calls < max_calls:
        optimizer.step()

        info = optimizer.last_step_info
        gradient_norms.append(info.get("grad_norm", None))
        posterior_variance_trace.append(info.get("sigma2", None))

        if mode == "original":
            f_values.append(optimizer.params_history_list[-1].clone())
            calls_at_iteration.append(objective._calls)
            inner_loop_samples.append(info.get("n_inner_samples", None))
        elif mode in ("argmax_pwolfe", "argmax_detwolfe", "ei_detwolfe"):
            inner_loop_samples.append(info.get("n_gi_samples_this_iter", None))
            n_steps_per_iter.append(info.get("n_steps_this_iter", None))
            alpha_history_per_iter.append(info.get("alpha_history", []))
            step_trigger_per_iter.append(info.get("step_trigger", None))
            if mode == "argmax_pwolfe":
                pwolfe_history_per_iter.append(info.get("p_wolfe_history", None))
            elif mode in ("argmax_detwolfe", "ei_detwolfe"):
                armijo_history_per_iter.append(info.get("armijo_history", None))
                curvature_history_per_iter.append(info.get("curvature_history", None)) 

            # entrys for call_at_iteration also for constrained while-loop behaviour
            trajectory = info.get("theta_trajectory")
            if trajectory:
                for theta_t, n_calls_t in trajectory:
                    f_values.append(theta_t if isinstance(theta_t, torch.Tensor) else torch.tensor(theta_t))
                    calls_at_iteration.append(n_calls_t)
            else:
                # Fallback to old single-checkpoint behaviour
                f_values.append(optimizer.params_history_list[-1].clone())
                calls_at_iteration.append(objective._calls)


    # Simple regret at each iteration instead of compute_rewards: f_max - best_so_far
    best_so_far = float("-inf")
    regret_per_eval = []
    for params_t in f_values:
        val = objective_raw(params_t.view(1, -1), observation_noise=False, requires_grad=False).item()
        best_so_far = max(best_so_far, val)
        regret_per_eval.append(float(f_max) - best_so_far)

    return {
        "regret_per_eval": regret_per_eval,
        "f_values": [p.numpy() for p in f_values],
        "inner_loop_samples": inner_loop_samples,
        "gradient_norms": gradient_norms,
        "posterior_variance_trace": posterior_variance_trace,
        "calls_at_iteration": calls_at_iteration,
        
        "n_steps_per_iter": n_steps_per_iter,
        "alpha_history_per_iter": alpha_history_per_iter,
        "step_trigger_per_iter": step_trigger_per_iter,
        "p_wolfe_history_per_iter": pwolfe_history_per_iter,
        "armijo_history_per_iter": armijo_history_per_iter,
        "curvature_history_per_iter": curvature_history_per_iter,
        
        "f_max": float(f_max),
        "config": cfg,
        "seed": seed,
        "dimension": dim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run thesis experiment for one variant."
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to pre-generated data directory.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load pre-generated data
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

    print(f"\nExperiment: {name}")
    print(f"Mode: {cfg['inner_loop_mode']}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs/dim: {n_runs}")
    print(f"Budget: {cfg['max_objective_calls']} objective calls\n")

    for dim in dimensions:
        dim_dir = os.path.join(results_dir, name, f"dim_{dim}")
        os.makedirs(dim_dir, exist_ok=True)

        print(f"dim={dim}:")
        for run_index in range(n_runs):
            seed = seed_start + run_index
            out_path = os.path.join(dim_dir, f"run_{run_index:03d}.pkl")

            if os.path.exists(out_path):
                print(f" run {run_index:03d} - already exists, skipping.")
                continue

            print(f" run {run_index:03d} (seed={seed}) ...", end=" ", flush=True)
            try:
                result = run_single(
                    cfg=cfg,
                    dim=dim,
                    seed=seed,
                    train_x=train_x_dict[dim][run_index],
                    train_y=train_y_dict[dim][run_index],
                    lengthscale=lengthscales_dict[dim],
                    f_max=f_max_dict[dim][run_index],
                )
                with open(out_path, "wb") as fh:
                    pickle.dump(result, fh)
                regret_final = result["regret_per_eval"][-1] if result["regret_per_eval"] else float("nan")
                n_iters = len(result["f_values"])
                print(f"done. iters={n_iters}, final_regret={regret_final:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                raise
        print()

    print(f"Results saved to: {os.path.join(results_dir, name)}")

if __name__ == "__main__":
    main()


