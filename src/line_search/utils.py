"""
Posterior evaluation utilities for line search termination criteria.

Mathematical conventions:
    theta: parameter shape: [1, D]
    p: search direction shape: [1, D]
    alpha: step size: scalar
    phi(a): 1D objective along p
    phi'(a): directional derivative

SE kernel (ARD):
    k(x1, x2) = sigma_f^2 exp(-1/2 sum_d (x1_d - x2_d)^2 / l_d^2)

Posterior kernel:
    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)
    where K_inv is the invertet Gram matrix
"""

from typing import Dict, Tuple
import torch

# prior kernel derivatives
def get_K_x1_x2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
   
    return model.covar_module(x1, x2).evaluate()


def get_K_x_X(model, x: torch.Tensor) -> torch.Tensor:
    
    X = model.train_inputs[0]
    return model.covar_module(x, X).evaluate()


def get_lengthscale_sq(model) -> torch.Tensor:
    
    return model.covar_module.base_kernel.lengthscale.detach().squeeze() ** 2


def get_dk_dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) for x2, shape [D]

    For the SE kernel:
        dk(x1, x2) /d x2_j = k(x1, x2) * (x1_i - x2_j)/ l_j^2
    """
    k = get_K_x1_x2(model, x1, x2).squeeze()         
    l2 = get_lengthscale_sq(model)                      
    diff = (x1 - x2).squeeze()                           
    return k * diff / l2                                  


def get_dk_dx1(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Analytic derivative of k(x1, x2) for x1, shape [D].

    For the SE kernel:
        dk(x1, x2) / d x1_i = -k(x1, x2) * (x1_i - x2_j) / l_j^2
                            = -dk(x1, x2) / d x2_j

    Sign follows from inner derivative of (x1_i -x2_j).
    """
    return -get_dk_dx2(model, x1, x2)                  


def get_d2k_dx1dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Mixed second derivative of k(x1, x2) for x1 and x2 shape [D, D].

    For the SE kernel:
        d^2k / (dx1_i  dx2_j) = k(x1,x2) * (delta{ij}/l_i^2
                                 - (x1_i-x2_i)*(x1_j-x2_j) / (l_i^2 * l_j^2))
    In matrix form:
        d^2k /(dx1  dx2^T) = k(x1,x2) *(diag(1/l^2) -outer(diff/l^2, diff/l^2))

    At x1 = x2 reduces to k(x,x) * diag(1/l^2) = sigma_f^2 * diag(1/l^2).
    """
    k = get_K_x1_x2(model, x1, x2).squeeze()           
    l2 = get_lengthscale_sq(model)                      
    diff = (x1 - x2).squeeze()                           
    diff_over_l2 = diff / l2                            
    return k * (torch.diag(1.0 / l2) - torch.outer(diff_over_l2, diff_over_l2))


# Posterior covariance and its analytic derivatives
def posterior_cov(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Posterior kernel k_post(x1, x2)
    
    k_post(x1, x2) = k(x1, x2) - k(x1, X) @ K_inv @ k(X, x2)

    At x1 = x2 this equals sigma2_post(x1), the posterior variance
    of the function.
    """
    K_inv = model.get_KXX_inv()                          
    k_x1_X = get_K_x_X(model, x1)                      
    k_X_x2 = get_K_x_X(model, x2).T                    
    k_prior = get_K_x1_x2(model, x1, x2)               
    return (k_prior - k_x1_X @ K_inv @ k_X_x2).squeeze()


def posterior_dcov_dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Derivative of k_post(x1, x2) for x2 shape [D].

    dk_post(x1, x2) / dx2
        = dk(x1, x2) / dx2 - k(x1, X) @ K_inv @ dk(X, x2) / dx2

    The matrix dk(X, x2) / dx2 has shape [N, D] where entry [j, d] is
        dk(X_j, x2) /dx2_d = k(X_j, x2) * (X_j_d - x2_d) / l_d^2

    This is what model._get_KxX_dx(x2) gives after reshaping:
        model._get_KxX_dx(x2) has shape [1, D, N]
        Entry [0, d, j] = -(x2_d - X_j_d)/l_d^2 * k(x2, X_j)
                        = (X_j_d - x2_d)/l_d^2 * k(X_j, x2) 
    so dk(X, x2) / dx2  [N, D] = model._get_KxX_dx(x2).squeeze(0).T
    """
    K_inv = model.get_KXX_inv()                         
    k_x1_X = get_K_x_X(model, x1)                    
    dk_prior = get_dk_dx2(model, x1, x2)                
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T                                   
    return dk_prior - (k_x1_X @ K_inv @ dk_X_x2).squeeze(0)  

# technically unused because of symmetry
def posterior_dcov_dx1(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Derivative of k_post(x1, x2) forx1, shape [D].

    dk_post(x1, x2) / dx1
        = dk(x1, x2 /dx1- dk(x1, X) / dx1 @ K_inv @ k(X, x2)

    The matrix dk(x1, X) / dx1 has shape [D, N] where entry [d, j] is
        dk(x1, X_j) / dx1_d

    This is model._get_KxX_dx(x1).squeeze(0), shape [D, N].

    Note: By symmetry of k_post,
        posterior_dcov_dx1(model, x0, xa) = posterior_dcov_dx2(model, xa, x0)
    Both formulations are used in different S-terms.
    """
    K_inv = model.get_KXX_inv()                       
    k_X_x2 = get_K_x_X(model, x2).T                   
    dk_prior = get_dk_dx1(model, x1, x2)                
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          
    return dk_prior - (dk_x1_X @ K_inv @ k_X_x2).squeeze() 


def posterior_d2cov_dx1dx2(model, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Mixed second derivative of k_post(x1, x2) for x1 and x2, shape [D, D].

    d^2k_post(x1, x2) / (dx1  dx2^T)
        = d^2k(x1, x2) / (dx1  dx2^T)- dk(x1, X) / dx1 @ K_inv @ dk(X, x2) / dx2

    Terms:
        dk(x1, X) / dx1 -->from model._get_KxX_dx(x1)
        K_inv --> from model.get_KXX_inv()             
        dk(X, x2) / dx2 -->from model._get_KxX_dx(x2).T
    

    At x1 = x2 = x this gives the posterior gradient covariance,
    which matches model.posterior_derivative(x)[1].
    """
    K_inv = model.get_KXX_inv()                          
    d2k_prior = get_d2k_dx1dx2(model, x1, x2)          
    dk_x1_X = model._get_KxX_dx(x1).squeeze(0)          
    dk_X_x2 = model._get_KxX_dx(x2).squeeze(0).T        
    return d2k_prior - dk_x1_X @ K_inv @ dk_X_x2        


# S-terms
def compute_s_terms(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
    sigma_floor: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """All cross-covariance terms for the probabilistic Wolfe conditions.
       Note: Exact S-Terms are given in Thesis.

       sigma_floor: Minimum posterior std as a fraction of sqrt(outputscale).
           Applied to diagonal S-terms to prevent pWolfe collapsing 
            near training data where variance --> 0.
            Design choice: due to different setting in comparison to Mahsereci & Hennigs Wiener process
            minimum variance, which never reaches zero along the path.
       Returns mapped Dict with Terms. 
    """
    x0 = theta                                  
    xa = theta + alpha * p                       
    p_vector = p.squeeze() #scalar                         

    with torch.no_grad():
        #,Diagonals
        S11 = model.posterior(x0).mvn.variance.squeeze()   
        S33 = model.posterior(xa).mvn.variance.squeeze()  

        _, var_d_0 = model.posterior_derivative(x0)       
        var_d_0 = var_d_0.squeeze()
        S22 = p_vector @ var_d_0 @ p_vector                     

        _, var_d_a = model.posterior_derivative(xa)        
        var_d_a = var_d_a.squeeze()
        S44 = p_vector @ var_d_a @ p_vector                    

        if sigma_floor > 0.0:
            outputscale = float(model.covar_module.outputscale.detach())
            floor_var = (sigma_floor ** 2) * outputscale
            ls_mean = model.covar_module.base_kernel.lengthscale.mean().item()
            floor_grad_var = floor_var / (ls_mean ** 2)
            S11 = S11.clamp(min=floor_var)
            S33 = S33.clamp(min=floor_var)
            S22 = S22.clamp(min=floor_grad_var)
            S44 = S44.clamp(min=floor_grad_var)


        # Cross-covariances between function values
        S13 = posterior_cov(model, x0, xa)                

        # Cross-covariances between function value and gradient
        S12 = p_vector @ posterior_dcov_dx2(model, x0, x0)   
        S14 = p_vector @ posterior_dcov_dx2(model, x0, xa)   
        S23 = p_vector @ posterior_dcov_dx2(model, xa, x0)   
        S34 = p_vector @ posterior_dcov_dx2(model, xa, xa)  

        # Cross-covariance between gradients
        d2cov = posterior_d2cov_dx1dx2(model, x0, xa)     
        S24 = p_vector @ d2cov @ p_vector                       

    return {
        'S11': S11, 'S22': S22, 'S33': S33, 'S44': S44,'S13': S13,
        'S12': S12, 'S14': S14, 'S23': S23, 'S24': S24,'S34': S34,
    }



# Main line search utils

def get_search_direction(model, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized search direction from posterior gradient mean.

    p = mean_d(theta) / ||mean_d(theta)||
    The search direction as normalized posterior gradient. (Euclidian)
    """
    with torch.no_grad():
        mean_d, _ = model.posterior_derivative(theta)   
        norm = mean_d.norm()
        if norm < 1e-10:
            # Degenerate: gradient is zero -->no meaningful direction
            p = torch.zeros_like(mean_d)
        else:
            p = mean_d / norm                           
    return p, mean_d # also raw search direction


def eval_phi(
    model,
    theta: torch.Tensor,
    alpha: float,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the 1D surrogate phi(alpha) = mu_post(theta + alpha*p).

    Returns: phi at alpha, directional derivative and variance at phi.
    """
    with torch.no_grad():
        x_new = theta + alpha * p                              

        posterior = model.posterior(x_new)
        phi = posterior.mvn.mean.squeeze()                     
        sigma2 = posterior.mvn.variance.squeeze()              

        mean_d, _ = model.posterior_derivative(x_new)         
        phi_d = (p * mean_d).sum()                         

    return phi, phi_d, sigma2


def eval_phi_0(
    model,
    theta: torch.Tensor,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation at alpha = 0 (for refrence point at beginning).
    """
    return eval_phi(model, theta, 0.0, p)


