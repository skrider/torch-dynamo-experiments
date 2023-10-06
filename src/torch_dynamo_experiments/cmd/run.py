import torch
import time

from torch_dynamo_experiments.util import pytorch_util as ptu
from torch_dynamo_experiments.util.util import Logger
from torch_dynamo_experiments.backend import backend_dict

import os

from torch_dynamo_experiments.util.util import timestamp

def profile_experiment(args, logdir_base, logger):
    model, example_inputs = (
        __import__(f"torchbenchmark.models.{args.model_name}")
        .models.__dict__[args.model_name]
        .Model(test="eval", device=ptu.device, batch_size=args.batch_size)
        .get_module()
    )
    model.eval()
    example_outputs = model(*example_inputs)
    __import__('pdb').set_trace()

    model = torch.compile(backend=backend_dict[args.backend])(model)

    if (args.capture_graph):
        logger.log_graph(model, example_inputs)

    def run_inference():
        model(*example_inputs)
    
    # check equality
    diff = torch.linalg.matrix_norm(model(*example_inputs) - example_outputs)
    logger.log_singleton_scalar(diff, "compiled_diff")

    print("Compiling model")
    start_time = time.time()
    run_inference()
    logger.log_singleton_scalar("compilation_time", time.time() - start_time)

    # profile model inference time
    ptu.profile_function(run_inference, logdir_base, "run", args.n_iter, logger, tensorboard=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=1000)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--capture_graph", type=bool, default=False)

    args = parser.parse_args()

    logdir = f"{args.logdir}/{args.model_name}_{args.backend}_{args.batch_size}_{args.n_iter}_{timestamp()}"

    if not (os.path.exists(args.logdir)):
        os.makedirs(args.logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # was getting a warning
    torch.set_float32_matmul_precision("high")

    logger = Logger(logdir)

    with torch.no_grad():
        profile_experiment(args, logdir, logger)

    logger.flush()


if __name__ == "__main__":
    main()
