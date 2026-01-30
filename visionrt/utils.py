from typing import Callable

import torch

from . import config


class logging:  # low effort logging class
    info: Callable = lambda x: (
        print(f"[visionrt] INFO: {x}") if config.verbose else None
    )
    warning: Callable = lambda x: (
        print(f"[visionrt] WARNING: {x}") if config.verbose else None
    )
    error: Callable = lambda x: (
        print(f"[visionrt] ERROR: {x}") if config.verbose else None
    )


def inference(model, ins, iters=500):  # get device times

    for _ in range(20):
        out = model(ins)

    torch.cuda.synchronize(0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(torch.cuda.Stream(0)):

        start.record()
        for _ in range(iters):
            out = model(ins)

        torch.cuda.synchronize(0)
        end.record()

    time_per_inference = start.elapsed_time(end) / iters

    out = model(ins).clone()
    return time_per_inference, out
