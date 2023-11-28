import torch
from typing import Callable

REGULARIZATION_FNS: dict[str, Callable] = {}
def regularization(reg_fn: Callable):
    REGULARIZATION_FNS[reg_fn.__name__] = reg_fn
    return reg_fn

@regularization
def hoyer_d_scaled_l1(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    hoyer_square_normalized = hoyer_square / acts.shape[-1]
    return hoyer_square_normalized * reg_coeff * acts.abs().sum()

@regularization
def l1(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    return reg_coeff * acts.abs().sum()

@regularization
def hoyer_square(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    l1_squared = acts.abs().sum(-1) ** 2
    l2_squared = (acts ** 2).sum(-1) + 1e-9
    squared = (l1_squared  / l2_squared).sum()
    # l1 = acts.norm(p=1, dim=-1)
    # l2 = acts.norm(p=2, dim=-1)
    # squared = ((l1 / l2)**2).sum()
    #print(squared, squared_old)
    return squared * reg_coeff

@regularization
def combined_hoyer_l1(acts: torch.Tensor, reg_coeffs: list[float]) -> list[torch.Tensor]:
    l1_coeff = reg_coeffs[0]
    hoyer_coeff = reg_coeffs[1]
    l1_loss = l1(acts, l1_coeff)
    hoyer_loss = hoyer_square(acts, hoyer_coeff)
    return [l1_loss, hoyer_loss]

@regularization
def combined_hoyer_sqrt(acts: torch.Tensor, reg_coeffs: list[float]) -> list[torch.Tensor]:
    l1_coeff = reg_coeffs[0]
    hoyer_coeff = reg_coeffs[1]
    l1_loss = sqrt(acts, l1_coeff)
    hoyer_loss = hoyer_square(acts, hoyer_coeff)
    return [l1_loss, hoyer_loss]

@regularization
def hoyer_d(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    '''from paper on signSGD'''
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    hoyer_square_normalized = hoyer_square / acts.shape[-1]
    return hoyer_square_normalized * reg_coeff


@regularization
def sqrt(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    act_reg_loss = acts.abs() + 1e-9
    act_reg_loss[(act_reg_loss < 1)] **= 0.5
    return reg_coeff * act_reg_loss.sum()
