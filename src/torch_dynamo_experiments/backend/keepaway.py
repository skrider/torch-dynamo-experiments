import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.node import Node
from torch.fx.graph import Graph
from typing import List, Dict
from torch_dynamo_experiments.util.pytorch_util import device


class Printer:
    def print(self, str):
        print("printing ", str)


def backend(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    modules: Dict[str, nn.Module] = dict(gm.named_modules())

    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = modules[node.target]
            with gm.graph.inserting_before(node):
                gm.graph.call_method("print", (Printer, mod._get_name()))

    gm.recompile()

    gm.forward(*example_inputs)

    return gm.forward
