import torch
from typing import Callable

REGULARIZATION_FNS: dict[str, Callable] = {}
def regularization(reg_fn: Callable):
    REGULARIZATION_FNS[reg_fn.__name__] = reg_fn
    return reg_fn


@regularization
def hoyer(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    l1 = acts.abs().sum()
    l2 = (acts ** 2).sum().sqrt()
    return reg_coeff * (l1 / l2)


@regularization
def l1(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    print(type(reg_coeff * acts.abs().sum()))
    return reg_coeff * acts.abs().sum()


@regularization
def sqrt(acts: torch.Tensor, reg_coeff: float) -> torch.Tensor:
    act_reg_loss = reg_coeff * acts.abs()
    act_reg_loss[(acts > 0) & (acts < 1)] **= 0.5
    return act_reg_loss.sum()
