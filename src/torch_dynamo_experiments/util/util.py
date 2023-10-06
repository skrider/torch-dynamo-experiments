from datetime import datetime
import os
from tensorboardX import SummaryWriter
import torch
import tempfile

def timestamp():
    presentDate = datetime.now()
    unix_timestamp = datetime.timestamp(presentDate) * 1000
    return str(int(unix_timestamp))

class Logger:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_singleton_scalar(self, scalar, name):
        self._summ_writer.add_scalar('{}'.format(name), scalar)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_graph(self, model: torch.nn.Module, example_inputs: torch.Tensor):
        # save as onnx
        path = tempfile.mktemp()
        torch.onnx.export(model, example_inputs, path, verbose=True)
        self._summ_writer.add_onnx_graph(path)
        os.remove(path)

    def flush(self):
        self._summ_writer.flush()

