import torch
import torch.nn.functional as F
from torch_dynamo_experiments.util.pytorch_util import device
from typing import Union, Callable


def make_experiment(backend: Union[str, Callable]) -> Callable:
    dim = 2**13

    parameter = 3 + torch.ones((dim)).to(device)

    def _run(input):
        return parameter + F.relu(
            input**2 + torch.normal(mean=input, std=1).to(device)
        )

    _run = torch.compile(backend=backend)(_run)

    input = torch.zeros((dim))
    input = torch.normal(mean=input, std=1).to(device)

    # call once to trigger compilation
    _run(input)

    return lambda: _run(input)
