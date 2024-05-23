import torch
from numbers import Number

def metrics_runtime_no_prefix(metrics, writer, epoch):
    for key in metrics.keys():
        if isinstance(metrics[key], Number):
            writer.add_scalar(f'{key}', metrics[key], epoch)
        elif torch.is_tensor(metrics[key]) and metrics[key].numel() == 1:
            writer.add_scalar(f'{key}', metrics[key].item(), epoch)
