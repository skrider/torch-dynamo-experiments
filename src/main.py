# AI
import torch
from transformers import BertTokenizer, BertModel
from datetime import datetime

def timestamp():
    presentDate = datetime.now()
    unix_timestamp = datetime.timestamp(presentDate)*1000
    return str(int(unix_timestamp))

device = "cuda"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode input text
input_text = "Hello, my dog is cute"
encoded_input = tokenizer(input_text, return_tensors='pt')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in evaluation mode to deactivate the DropOut modules
model.eval()

# If you have a GPU, put everything on cuda
encoded_input = encoded_input.to(device)
model.to(device)

N = 1000

stack_file_path = f"/data/profile_stack_trace_{timestamp()}"
# stack_file = open(stack_file_path, "w")
# stack_file.write("")
# stack_file.close()

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    print(f"Writing stack to {stack_file_path}")
    prof.export_stacks(stack_file_path, metric="self_cuda_time_total")

print("starting profile...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler,
    with_stack=True
    ) as p:
    for iter in range(N):
        outputs = model(**encoded_input)
        p.step()

