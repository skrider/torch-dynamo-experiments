import torch
from typing import List


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    __import__("pdb").set_trace()
    return gm.forward
