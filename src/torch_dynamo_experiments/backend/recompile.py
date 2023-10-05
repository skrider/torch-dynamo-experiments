import torch
from typing import List
import tempfile
import importlib
import torch_dynamo_experiments.util.pytorch_util as ptu


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    dtype = next(gm.parameters()).dtype
    tmpdir = tempfile.mkdtemp()
    gm.to_folder(tmpdir, "Model")

    # AI
    spec = importlib.util.spec_from_file_location("model", f"{tmpdir}/module.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.Model().to(ptu.device)
    if dtype == torch.float16:
        model.half()
    return model.forward
