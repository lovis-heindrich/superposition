import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Callable
from regularization import REGULARIZATION_FNS

class AutoEncoder(nn.Module):
    def __init__(
        self, d_hidden: int, reg_coeff: float | list[float], d_in: int, dtype=torch.float32, seed=47, reg: Literal["l1", "sqrt", "hoyer"] = "l1"
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_hidden, dtype=dtype))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_in, dtype=dtype))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.reg_coeff = reg_coeff
        self.reg = reg

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
        reg_losses = REGULARIZATION_FNS[self.reg](acts, self.reg_coeff)
        mse_loss = loss.clone()
        if isinstance(reg_losses, list) or isinstance(reg_losses, tuple):
            for reg_loss in reg_losses:
                loss += reg_loss
        else:
            loss += reg_losses
        return loss, x_reconstruct, acts, mse_loss, reg_losses

    @torch.no_grad()
    def norm_decoder(self):
        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
    
    # @torch.no_grad()
    # def make_decoder_weights_and_grad_unit_norm(self):
    #     W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
    #     W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
    #     self.W_dec.grad -= W_dec_grad_proj
    #     # Bugfix(?) for ensuring W_dec retains unit norm
    #     self.W_dec.data = W_dec_normed
    

