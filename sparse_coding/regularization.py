import torch
from typing import Callable

REGULARIZATION_FNS: dict[str, Callable] = {}
def regularization(reg_fn: Callable):
    REGULARIZATION_FNS[reg_fn.__name__] = reg_fn
    return reg_fn

@regularization
def l1(acts: torch.Tensor, scale: float) -> torch.Tensor:
    if isinstance(scale, list) or isinstance(scale, tuple):
        scale = scale[0]
    return scale * acts.abs().sum()

@regularization
def sqrt(acts: torch.Tensor, scale: float | list[float]) -> torch.Tensor:
    '''Square root penalty in [0, 1] and L1 elsewhere'''
    if isinstance(scale, list) or isinstance(scale, tuple):
        scale = scale[0]

    l1 = acts.abs()
    mask = l1 < 1
    loss = (l1.sqrt() * mask) + (l1 * ~mask)
    return loss.sum() * scale

@regularization
def hoyer_square(acts: torch.Tensor, scale: float, dim: int = -1) -> torch.Tensor:
    """Hoyer-Square sparsity measure."""
    if isinstance(scale, list) or isinstance(scale, tuple):
        scale = scale[0]

    eps = torch.finfo(acts.dtype).eps

    numer = acts.norm(p=1, dim=dim)
    denom = acts.norm(p=2, dim=dim)
    return numer.div(denom + eps).square().sum() * scale

@regularization
def combined_hoyer_l1(acts: torch.Tensor, reg_coeffs: list[float]) -> list[torch.Tensor]:
    l1_coeff, hoyer_coeff = reg_coeffs[0], reg_coeffs[1]

    l1_loss = l1(acts, l1_coeff)
    hoyer_loss = hoyer_square(acts, hoyer_coeff)
    return [l1_loss, hoyer_loss]

@regularization
def combined_hoyer_sqrt(acts: torch.Tensor, reg_coeffs: list[float]) -> list[torch.Tensor]:
    sqrt_coeff, hoyer_coeff = reg_coeffs[0], reg_coeffs[1]

    sqrt_loss = sqrt(acts, sqrt_coeff)
    hoyer_loss = hoyer_square(acts, hoyer_coeff)
    return [sqrt_loss, hoyer_loss]

# Just for reference
def hoyer_sqrt_reg(acts: torch.Tensor, hoyer_coeff: float, l1_coeff: float):
    # Hoyer squared loss
    l1_squared = acts.abs().sum(-1) ** 2
    l2_squared = (acts ** 2).sum(-1) + 1e-9
    hoyer_squared = (l1_squared  / l2_squared).sum()
    # L1 loss with sqrt for activations < 1
    act_reg_loss = acts.abs() + 1e-9
    mask = act_reg_loss < 1
    l1_sqrt = act_reg_loss * ~mask 
    l1_sqrt += (act_reg_loss * mask) ** 0.5
    return hoyer_coeff * hoyer_squared + l1_coeff * l1_sqrt.sum()

@regularization
def hoyer_d_scaled_l1(acts: torch.Tensor, scale: float) -> torch.Tensor:
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    hoyer_square_normalized = hoyer_square / acts.shape[-1]
    return hoyer_square_normalized * scale * acts.abs().sum()

@regularization
def hoyer_d(acts: torch.Tensor, scale: float) -> torch.Tensor:
    '''from paper on signSGD'''
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    hoyer_square_normalized = hoyer_square / acts.shape[-1]
    return hoyer_square_normalized * scale