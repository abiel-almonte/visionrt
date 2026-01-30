import os
import copy
from typing import List, Dict, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

torch.set_float32_matmul_precision("high")

from torch._dynamo import register_backend
from torch._inductor.compile_fx import compile_fx

import torch._inductor.config

torch._inductor.config.debug = True  # dump code-gen
# torch._inductor.config.triton.cudagraphs = True # triton cuda graph capture, messes with mine

from ._visionrt import GraphExecutor
from .utils import inference, logging
from .optim import (
    Placeholder,
    Transformation,
    optimize_fx,
    inspect_fx,
    transformation_registry,
)
from . import config


class LazyGraphExecutor:
    def __init__(self, fn):
        self._fn = fn
        self._ge = GraphExecutor(fn)
        self._failed = False

    def is_captured(self):
        return self._ge.is_captured()

    def __call__(self, ins):
        if self._failed:
            return self._fn(ins)

        if not self.is_captured():  # capture the cuda graph lazily
            try:
                self._ge.capture(ins)
                logging.info("Captured cudagraph lazily")
            except Exception as e:
                logging.warning(
                    f"cudagraph capture failed: {e}\nFalling back to compiled model."
                )
                self._failed = True
                return self._fn(ins)

        return self._ge(ins)


class CompiledModel:
    def __init__(self, caller, cfg):
        self._caller = caller
        self._cfg = cfg
        self._initialized = True

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError("CompiledModel is immutable")

        super().__setattr__(name, value)

    @property
    def config(self):
        return self._cfg

    def __call__(self, *args):
        return self._caller(*args)

    def __repr__(self):
        cudagraph_status = "disabled"

        if isinstance(self._caller, LazyGraphExecutor):
            if self._caller._failed:
                cudagraph_status = "capture failed"
            elif self._caller.is_captured():
                cudagraph_status = "captured"
            else:
                cudagraph_status = "not captured"

        optims = self._cfg.get("custom_optims", [])
        optims_str = ", ".join(optims) if optims else "none"

        parts = [
            f"cudagraph={cudagraph_status}",
            f"custom_optims=[{optims_str}]",
            f"inductor={self._cfg.get('use_inductor', True)}",
        ]

        return f"CompiledModel({', '.join(parts)})"


@register_backend  # type:ignore
def visionrt(gm: fx.GraphModule, ins):

    if config.use_custom:

        placeholders: Dict[str, Placeholder] = {
            str(node.target): Placeholder(
                value=input,
                is_parameter=isinstance(input, nn.Parameter),
            )
            for node, input in zip(gm.graph.nodes, ins)
            if node.op == "placeholder"  # defensive
        }

        xforms: List[Tuple[str, Transformation]] = [
            (name, xform)
            for name, xform in transformation_registry.items()
            if getattr(config.optims, name, False)
        ]

        logging.info(
            f"Applying {len(xforms)} transformations: {', '.join(name for name, _ in xforms)}"
        )

        gm, ins = optimize_fx(
            gm=gm,
            placeholders=placeholders,
            transformations=[xform for _, xform in xforms],
        )

    else:
        logging.info(
            "custom transformations disabled. Enable it with a custom optim in `config.optims.<xform>`"
        )

    # inductor_config.freezing = True  # no more constant folding, but ruins config.triton.cudagraphs performance
    inspect_fx(gm, ins)

    if config.use_inductor:
        return compile_fx(
            model_=gm, example_inputs_=ins
        )  # route model to inductor after custom backend
    else:
        logging.info("inductor backend disabled. Enable it with `config.use_inductor`")
        return gm


def compile(model: nn.Module):
    cfg = {
        "use_inductor": config.use_inductor,
        "cudagraphs": config.cudagraphs,
        "custom_optims": [
            xform
            for xform in config.optims.__annotations__
            if getattr(config.optims, xform, False)
        ],
    }

    fn = torch.compile(model, backend="visionrt", dynamic=False)
    caller = LazyGraphExecutor(fn) if config.cudagraphs else fn

    return CompiledModel(caller, cfg)


# Find the generated code here:
# GENERATE_BASELINE_CODE=0 TORCH_LOGS="output_code" python3 compiler.py 2>&1 | grep -E "Output code written to:|CUDA Time (ms):"

if __name__ == "__main__":
    GENERATE_BASELINE_CODE = int(os.environ.get("GENERATE_BASELINE_CODE", "1"))
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).cuda().eval()
    input = torch.randn(1, 3, 224, 224).cuda()

    if GENERATE_BASELINE_CODE:
        torch_compiled_model = torch.compile(
            copy.deepcopy(
                model
            ),  # copy model incase the backend modifies the model inplace
            backend="inductor",
            dynamic=False,
        )

    torch._dynamo.reset()
    config.use_inductor = True
    config.debug = True
    config.verbose = True

    config.cudagraphs = True
    config.optims.fold_conv_bn = True

    vrt_compiled_model = compile(copy.deepcopy(model))

    with torch.inference_mode():

        baseline_time, baseline_out = inference(model, input)
        if GENERATE_BASELINE_CODE:
            torch_compiled_time, torch_compiled_out = inference(
                torch_compiled_model, input
            )  # type:ignore
        vrt_compiled_time, vrt_compiled_out = inference(vrt_compiled_model, input)

        print(f"Baseline CUDA Time (ms): {baseline_time}")
        if GENERATE_BASELINE_CODE:
            print(
                f"Torch Compiled CUDA Time (ms): {torch_compiled_time}"
            )  # type:ignore
        print(f"VRT Compiled CUDA Time (ms): {vrt_compiled_time}")

        print()

        if GENERATE_BASELINE_CODE:
            print(
                f"Torch Max diff: {(baseline_out - torch_compiled_out.clone()).abs().max()}"
            )  # type:ignore
        print(f"VRT Max diff: {(baseline_out - vrt_compiled_out).abs().max()}")
