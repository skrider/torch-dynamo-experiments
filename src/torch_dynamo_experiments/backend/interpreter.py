import torch
import torch.fx
from typing import List


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    i = torch.fx.Interpreter(gm)
    return i.run
