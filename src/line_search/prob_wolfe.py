from typing import Tuple
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm as scipy_norm

from src.line_search.utils import (
    eval_phi,
    eval_phi_0,
    compute_s_terms,
    get_search_direction,
)
# dead
def find_alpha_star(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    delta: float = 0.2,
) -> float:
    """Find alpha* = argmax_{alpha in (0, delta]} mu_post(theta + alpha*p).
    """
    def negative_phi(alpha: float) -> float:
        phi, _, _ = eval_phi(model, theta, alpha, p)
        return -phi.item()

    result = minimize_scalar(
        negative_phi,
        bounds=(1e-6, delta),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 500},
    )
    return float(result.x)

def compute_pwolfe(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    phi_0: torch.Tensor = None,
    phi_d_0: torch.Tensor = None,
    c1: float = 0.05,
    c2: float = 0.5,
    sigma_floor: float = 0.0,
) -> float:
    """Compute p_Wolfe(alpha) = P(Both conditions satisfied | GP posterior).
     
       Exact derivation in thesis.
    """
    # values at alpha=0 
    if phi_0 is None or phi_d_0 is None:
        phi_0, phi_d_0, _ = eval_phi_0(model, theta, p)

    phi_a, phi_d_a, _ = eval_phi(model, theta, alpha, p)

    # cross-covariance terms
    s = compute_s_terms(model, theta, alpha, p, sigma_floor=sigma_floor)

    # Means of a_t (Armijo) and b_t (curvature) 
    m_a = phi_a - phi_0 - c1 * alpha * phi_d_0
    m_b = c2 * phi_d_0 - phi_d_a

    # C_aa, C_bb and C_ab 
    C_aa = (
        s['S11'] + s['S33'] - 2 * s['S13'] + (c1 * alpha) ** 2 * s['S22']
        + 2 * c1 * alpha * s['S12']- 2 * c1 * alpha * s['S23']
    )
    C_bb = s['S44'] + c2 ** 2 * s['S22'] - 2 * c2 * s['S24']

    C_ab = (
        -c2 * s['S12'] + s['S14']- c1 * alpha * c2 * s['S22']
        + c1 * alpha * s['S24'] + c2 * s['S23'] - s['S34']
    )

    # Numerical stability
    C_aa = C_aa.clamp(min=1e-10)
    C_bb = C_bb.clamp(min=1e-10)
    sqrt_Caa = C_aa.sqrt()
    sqrt_Cbb = C_bb.sqrt()

    # Correlation coefficient
    rho = (C_ab / (sqrt_Caa * sqrt_Cbb)).clamp(-1 + 1e-6, 1 - 1e-6)
    rho_value = rho.item()

    # Standardized limits 
    h_a = (-m_a / sqrt_Caa).item()
    h_b = (-m_b / sqrt_Cbb).item()

    # Upper bound for curvature condition (strong 95% confidence from paper):
    b_bar = 2 * c2 * (phi_d_0 + 2 * s['S22'].sqrt())
    upperbound = ((b_bar - m_b) / sqrt_Cbb).item()

    # Guard against empty integration interval
    if upperbound <= h_b:
        return 0.0
    
    # Full Decomposition in Thesis.
     
    # Decomposition:
    #    p_Wolfe = Phi(upb) - Phi(h_b) - Phi2(h_a, upb; rho) + Phi2(h_a, h_b; rho)
    cov_matrix = [[1.0, rho_value], [rho_value, 1.0]]
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix)

    pwolfe = (
        scipy_norm.cdf(upperbound) - scipy_norm.cdf(h_b) - mvn.cdf([h_a, upperbound]) + mvn.cdf([h_a, h_b])
    )

    return float(max(0.0, pwolfe))


def check_prob_wolfe(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
    phi_0: torch.Tensor,
    phi_d_0: torch.Tensor,
    delta: float = 0.2,
    c1: float = 0.05,
    c2: float = 0.5,
    c_W: float = 0.3,
) -> Tuple[bool, float, float]:
    
    alpha_candidate = find_alpha_star(model, theta, p, delta)
    pwolfe_value = compute_pwolfe(
                    model, theta, alpha_candidate, p,
                    phi_0=phi_0, phi_d_0=phi_d_0,
                    c1=c1, c2=c2,
                 )
    wolfe_satisfied = pwolfe_value > c_W
    return wolfe_satisfied, alpha_candidate, pwolfe_value

