import os
import sys
import torch
sys.path.append("../")  # Add the parent directory to the system path
from utils.haystack_utils import get_device


def test_wikipedia_data():
    if os.path.isfile(f"data/wikipedia/de_batched_0.pt"):
        data = torch.load(f"data/wikipedia/de_batched_0.pt")
        assert len(data.shape) == 2
        assert type(data[0, 0].item()) == int

def test_europarl_data():
    if os.path.isfile(f"data/europarl/de_batched.pt"):
        data = torch.load(f"data/europarl/de_batched.pt")
        assert len(data.shape) == 2
        assert type(data[0, 0].item()) == int


test_wikipedia_data()
test_europarl_data()

