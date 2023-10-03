import torch
import time

from torch_dynamo_experiments.util import pytorch_util as ptu
from torch_dynamo_experiments.backend import backend_dict

import os

from torch_dynamo_experiments.util.util import timestamp

def profile_experiment(args, logdir_base):
    model_import = __import__(f"torchbenchmark.models.{args.model_name}")
    model, example_inputs = (
        model_import.models.__dict__[args.model_name]
        .Model(test="eval", device="cpu", batch_size=args.batch_size)
        .get_module()
    )
    model.eval()

    # profile cuda load time
    def load_unload_model():
        model.to(ptu.device)
        model.to("cpu")
    logdir = f"{logdir_base}/load"
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ptu.profile_function(load_unload_model, logdir, 10)

    model.to(ptu.device)
    example_inputs = [i.to(ptu.device) for i in example_inputs if type(i) == torch.Tensor]

    model = torch.compile(backend=backend_dict[args.backend])(model)
    # profile model compile time
    def run_inference():
        model(*example_inputs)
    # logdir = f"{args.logdir}/compile"
    # if not (os.path.exists(logdir)):
    #     os.makedirs(logdir)
    # 
    # # TODO find out a way to invalidate the cache and compile multiple times
    # ptu.profile_function(run_inference, logdir, 1, warmup=False)
    print("Compiling model")
    start_time = time.time()
    run_inference()
    end_time = time.time()
    print(f"Compiling model took {end_time - start_time} seconds")

    # profile model inference time
    logdir = f"{logdir_base}/run"
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ptu.profile_function(run_inference, logdir, args.n_iter, tensorboard=True)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=1000)
    parser.add_argument("--batch_size", "-b", type=int, default=1)

    args = parser.parse_args()

    logdir = f"{args.logdir}/{args.model_name}_{args.backend}_{args.batch_size}_{timestamp()}"

    if not (os.path.exists(args.logdir)):
        os.makedirs(args.logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # was getting a warning
    torch.set_float32_matmul_precision("high")
    
    with torch.no_grad():
        profile_experiment(args, logdir)


if __name__ == "__main__":
    main()
