import torch
from typing import List

def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return gm.forward
