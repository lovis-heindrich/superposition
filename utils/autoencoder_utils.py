import torch.nn.functional as F
from jaxtyping import Int, Float
from torch import Tensor

import sys
sys.path.append('../')  # Add the parent directory to the system path
import utils.haystack_utils as haystack_utils
from sparse_coding.train_autoencoder import AutoEncoder


def custom_forward(enc: AutoEncoder, x: Float[Tensor, "batch d_in"], neuron: int, activation: float):
    x_cent = x - enc.b_dec
    acts = F.relu(x_cent @ enc.W_enc + enc.b_enc)
    acts[:, neuron] = activation
    x_reconstruct = acts @ enc.W_dec + enc.b_dec
    l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
    l1_loss = enc.l1_coeff * (acts.abs().sum())
    loss = l2_loss + l1_loss
    return loss, x_reconstruct, acts, l2_loss, l1_loss