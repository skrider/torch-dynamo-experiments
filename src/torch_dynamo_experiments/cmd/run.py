import torch

from torch_dynamo_experiments.util import pytorch_util as ptu
from torch_dynamo_experiments.util import util
from torch_dynamo_experiments.experiment import experiment_dict
from torch_dynamo_experiments.backend import backend_dict

WAIT_STEPS = 1
WARMUP_STEPS = 1
REPEAT = 1

def profile_experiment(args):
    stack_file_path = f"{args.logdir}/{args.activity}_{args.backend}_{args.exp_name}_{util.timestamp()}"
    make_exp = experiment_dict[args.exp_name]
    experiment_fn = make_exp(backend_dict[args.backend])

    def trace_handler(prof):
        torch.profiler.tensorboard_trace_handler(args.logdir)(prof)
        with open(f"{stack_file_path}.txt", "w") as f:
            f.write(p.key_averages().table(sort_by=f"self_{args.activity}_time_total", row_limit=10))
        print(f"Writing stack to {stack_file_path}")
        prof.export_stacks(stack_file_path, metric=f"self_{args.activity}_time_total")
        print("done export stack")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            ],
        schedule=torch.profiler.schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=args.n_iter,
            repeat=1),
        on_trace_ready=trace_handler,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        with_stack=True
        ) as p:
        print("running experiment")
        for _ in range(WAIT_STEPS + WARMUP_STEPS + args.n_iter + 1):
            experiment_fn()
            p.step()
        print("experiment complete")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--activity", type=str, default="cpu")
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--logdir", type=str, default="/data")
    parser.add_argument("--n_iter", "-n", type=int, default=1000)

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    profile_experiment(args)

if __name__ == "__main__":
    main()

