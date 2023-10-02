import torch
import random

from torch_dynamo_experiments.util import pytorch_util as ptu
from torch_dynamo_experiments.util import util
from torch_dynamo_experiments.experiment import experiment_dict
from torch_dynamo_experiments.backend import backend_dict
from huggingface_hub import HfFileSystem

import os

WAIT_STEPS = 1
WARMUP_STEPS = 1
REPEAT = 1

HF_TOKEN = os.environ["HF_TOKEN"]

def download_model(out_dir: str, model_id: str) -> str:
    hf_path = f"{model_id}/pytorch_model.bin"
    out_file = f"{out_dir}/{str(hf_path).replace('/', '--')}"

    if os.path.isfile(out_file):
        print(f"Skipping download already present {out_file}")
    else:
        out_temp = f"{out_dir}/{random.randint(0, 2**31)}"
        print(f"Downloading {hf_path} to {out_temp}")
        HfFileSystem(token=HF_TOKEN).download(hf_path, out_temp)
        print(f"Atomically renaming {out_temp} to {out_file}")
        os.rename(out_temp, out_file)

    return out_file

def profile_experiment(args):
    model_out = download_model(args.model_cache_dir, args.model_name)

    model = torch.load(model_out)

    __import__('pdb').set_trace()

    # profile cuda load time
    def load_unload_model():
        model.to(ptu.device)
        model.to("cpu")

    ptu.profile_function(load_unload_model, args.logdir, 10, ["cpu", "cuda"])

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_cache_dir", type=str, default="/tmp/models")
    parser.add_argument("--activity", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=1000)

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    profile_experiment(args)


if __name__ == "__main__":
    main()
