import torch
from typing import Callable

REGULARIZATION_FNS: dict[str, Callable] = {}
def regularization(reg_fn: Callable):
    REGULARIZATION_FNS[reg_fn.__name__] = reg_fn
    return reg_fn


@regularization
def hoyer(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    '''hoyer square'''
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    return hoyer_square * reg_coeff


@regularization
def hoyer_d(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    '''from paper on signSGD'''
    l1_squared = acts.abs().sum() ** 2
    l2_squared = (acts ** 2).sum()
    hoyer_square = l1_squared  / l2_squared
    hoyer_square_normalized = hoyer_square / acts.shape[-1]
    return hoyer_square_normalized * reg_coeff


@regularization
def l1(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    print(type(reg_coeff * acts.abs().sum()))
    return reg_coeff * acts.abs().sum()


@regularization
def sqrt(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    act_reg_loss = reg_coeff * acts.abs()
    act_reg_loss[(acts > 0) & (acts < 1)] **= 0.5
    return act_reg_loss.sum()
