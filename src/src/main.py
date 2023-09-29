import torch
from transformers import BertTokenizer, BertModel
from datetime import datetime
from typing import List

device = "cuda"

def timestamp():
    presentDate = datetime.now()
    unix_timestamp = datetime.timestamp(presentDate)*1000
    return str(int(unix_timestamp))

def tracing_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode input text
input_text = "Hello, my dog is cute"
encoded_input = tokenizer(input_text, return_tensors='pt')
encoded_input = encoded_input['input_ids']

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in evaluation mode to deactivate the DropOut modules
model.eval()

# If you have a GPU, put everything on cuda
encoded_input = encoded_input.to(device)
model.to(device)

@torch.compile(backend=tracing_backend)
def run_noisy_inference(input):
    return model(input)

N = 1000

stack_file_path = f"/data/profile_stack_trace_{timestamp()}"

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    print(f"Writing stack to {stack_file_path}")
    prof.export_stacks(stack_file_path, metric="self_cpu_time_total")
    print("done export stack")

# print("starting profile...")
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#         ],
#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=2,
#         repeat=1),
#     on_trace_ready=trace_handler,
#     experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
#     with_stack=True
#     ) as p:
for iter in range(N):
    outputs = run_noisy_inference(encoded_input)
#     p.step()

