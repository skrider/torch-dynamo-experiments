import torch
import time
import os
from typing import Callable, List
from torch_dynamo_experiments.util.util import timestamp, Logger

device = "cuda"

WAIT_STEPS = 1
WARMUP_STEPS = 1
REPEAT = 1


def profile_function(
    fn: Callable,
    out_base_dir: str,
    name: str,
    n: int,
    logger: Logger,
    activities: List[str] = ["cpu", "cuda"],
    tensorboard: bool = False,
    warmup: bool = True,
):
    out_dir = f"{out_base_dir}/{name}"
    if not (os.path.exists(out_dir)):
        os.makedirs(out_dir)

    def trace_handler(prof):
        if tensorboard:
            torch.profiler.tensorboard_trace_handler(out_dir)(prof)

        with open(f"{out_dir}/table.txt", "w") as f:
            f.write(
                p.key_averages().table(sort_by=f"self_cpu_time_total", row_limit=100)
            )

        for a in activities:
            stack_file_path = f"{out_dir}/{a}.prof"
            print(f"Writing {a} stack to {stack_file_path}")
            prof.export_stacks(stack_file_path, metric=f"self_{a}_time_total")
            print("done export stack")

    a = []
    if "cuda" in activities:
        a.append(torch.profiler.ProfilerActivity.CUDA)
    if "cpu" in activities:
        a.append(torch.profiler.ProfilerActivity.CPU)

    wait_steps = WAIT_STEPS if warmup else 0
    warmup_steps = WARMUP_STEPS if warmup else 0

    mean_time = 0

    with torch.profiler.profile(
        activities=a,
        schedule=torch.profiler.schedule(
            wait=wait_steps, warmup=warmup_steps, active=n, repeat=1
        ),
        profile_memory=True,
        on_trace_ready=trace_handler,
        # have to include this or else cuda stacks don't get exported, see
        # https://github.com/pytorch/pytorch/issues/100253
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        with_stack=True,
    ) as p:
        print("running experiment")
        for i in range(WAIT_STEPS + WARMUP_STEPS + n + 1):
            time_start = time.time()
            fn()
            p.step()
            time_end = time.time()
            if i >= WAIT_STEPS + WARMUP_STEPS:
                mean_time += time_end - time_start
            logger.log_scalar(time_end - time_start, f"{name}_time", i)
        print("experiment complete")

    logger.log_singleton_scalar(mean_time / n, f"{name}_mean_time")
