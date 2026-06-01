import torch
from src.line_search.utils import eval_phi
from src.line_search.det_ei import compute_ei, check_wolfe
from src.line_search.prob_wolfe import compute_pwolfe


def find_alpha_constrained(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_d_0: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    mode: str,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
    sigma_floor: float = 0.0,
    N: int = 50,
):
    """Grid search for the best feasible alpha.

       Returns: best feasible alpha for inner loop mode and cached values for used constraint. (Metrics)
    """
    alphas = torch.linspace(alpha_min, alpha_max, N).tolist()

    best_alpha = None
    best_objective = -float('inf')
    best_constraint = None  

    for alpha in alphas:
        if mode == 'argmax_pwolfe':
            pWolfe = compute_pwolfe(
                model, theta, alpha, p,
                phi_0=phi_0, phi_d_0=phi_d_0,
                c1=c1, c2=c2, sigma_floor=sigma_floor,
            )
            feasible = pWolfe > c_W
            constraint_value = pWolfe

        else:
            armijo_ok, curvature_ok = check_wolfe(
                model, theta, alpha, p,
                phi_0=phi_0, phi_d_0=phi_d_0,
                c1=c1, c2=c2,
            )
            feasible = armijo_ok and curvature_ok
            constraint_value = (armijo_ok, curvature_ok)

        if not feasible:
            continue

        if mode in ('argmax_pwolfe', 'argmax_detwolfe'):
            phi, _, _ = eval_phi(model, theta, alpha, p)
            objective = phi.item()
        else:
            objective = compute_ei(model, theta, alpha, p, eta=phi_0)

        if objective > best_objective:
            best_objective = objective
            best_alpha = alpha
            best_constraint = constraint_value

    return best_alpha, best_constraint
