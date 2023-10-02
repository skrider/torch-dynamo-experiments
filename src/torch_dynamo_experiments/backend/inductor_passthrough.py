import torch
from typing import List

def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    __import__('pdb').set_trace()
    compiled = torch._inductor.compile(gm, example_inputs)
    return compiled

