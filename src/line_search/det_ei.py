from typing import Tuple
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import norm as scipy_norm

from src.line_search.utils import (
    eval_phi, eval_phi_0, get_search_direction,
)
# Import compute_p_wolfe for ei_pwolfe combined check --> old ablation
from src.line_search.prob_wolfe import compute_pwolfe

def compute_ei(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    eta: torch.Tensor,
) -> float:
    """Compute Expected Improvement EI(alpha) for maximization.

    EI(alpha) = (mu - eta) * Phi(z) + sigma * phi_pdf(z)
    where z = (mu - eta) / sigma, mu = mu_post(theta + alpha*p),
    sigma = sqrt(sigma2_post(theta + alpha*p))
    """
    phi, _, sigma2 = eval_phi(model, theta, alpha, p)
    sigma = sigma2.clamp(min=0.0).sqrt()
    mu_val = phi.item()
    sigma_val = sigma.item()
    eta_val = eta.item()

    if sigma_val < 1e-10:
        #GP fully confident:EI = max(mu - eta, 0). only exploitation
        return float(max(mu_val - eta_val, 0.0))

    z = (mu_val - eta_val) / sigma_val
    ei = (mu_val - eta_val) * scipy_norm.cdf(z) + sigma_val * scipy_norm.pdf(z)
    return float(max(ei, 0.0))

# dead code
def find_best_alpha_ei(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
) -> float:
   
    def negative_ei(alpha: float) -> float:
        return -compute_ei(model, theta, alpha, p, eta)

    result = minimize_scalar(
        negative_ei,
        bounds=(1e-6, delta),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 500},
    )
    return float(result.x)


def check_wolfe(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_d_0: torch.Tensor,
    c1: float = 0.05,
    c2: float = 0.5,
) -> Tuple[bool, bool]:
 
    phi_alpha, phi_d_a, _ = eval_phi(model, theta, alpha, p)

    armijo_ok = bool(
        phi_alpha >= phi_0 + c1 * alpha * phi_d_0
    )
    curvature_ok = bool(
        phi_d_a.abs() <= c2 * phi_d_0.abs()
    )
    return armijo_ok, curvature_ok


# dead
def check_wolfe_combined(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_d_0: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
) -> Tuple[bool, float, bool, bool]:
   
    alpha_candidate = find_best_alpha_ei(model, theta, p, eta, delta)
    armijo_ok, curvature_ok = check_wolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_d_0=phi_d_0,
        c1=c1, c2=c2,
    )
    wolfe_satisfied = armijo_ok and curvature_ok
    return wolfe_satisfied, alpha_candidate, armijo_ok, curvature_ok

#combined with Variant A (dead)
def check_ei_pwolfe(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_d_0: torch.Tensor,
    eta: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
    sigma_floor: float = 0.1,
) -> Tuple[bool, float, float, bool, bool]:
   
    alpha_candidate = find_best_alpha_ei(model, theta, p, eta, delta=delta)

    p_wolfe_value = compute_pwolfe(
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_d_0=phi_d_0,
        c1=c1, c2=c2, sigma_floor=sigma_floor,
    )
# Metrics for pwolfe case
    armijo_ok, curvature_ok = check_wolfe( 
        model, theta, alpha_candidate, p,
        phi_0=phi_0, phi_d_0=phi_d_0,
        c1=c1, c2=c2,
    )

    wolfe_satisfied = p_wolfe_value > c_W
    return wolfe_satisfied, alpha_candidate, p_wolfe_value, armijo_ok, curvature_ok
