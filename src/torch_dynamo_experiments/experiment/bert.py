import torch
from transformers import BertModel
from typing import Union, Callable
from torch_dynamo_experiments.util.pytorch_util import device


def make_experiment(backend: Union[str, Callable]):
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(device)

    def _run(input):
        return model(input)

    _run = torch.compile(backend=backend)(_run)

    input = torch.randint(0, 10000, (512, 1)).to(device)
    _run(input)

    return lambda: _run(input)
