from dataclasses import dataclass
from typing import Protocol, List, Dict
import operator

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.lowering import make_fallback

import triton
import triton.language as tl

#from ._visionrt import fused_add_relu_cuda
from . import config


@dataclass
class Placeholder:
    value: torch.Tensor
    is_parameter: bool  # TODO: give this type instead of bool (value_type)


class Transformation(Protocol):
    def __call__(
        self, gm: fx.GraphModule, ph_dict: Dict[str, Placeholder], node: fx.Node
    ) -> None: ...


transformation_registry: Dict[str, Transformation] = {}


def register_transformation(config_name: str):

    def decorator(fn: Transformation) -> Transformation:
        transformation_registry[config_name] = fn
        return fn

    return decorator


@register_transformation("fold_conv_bn")
def fold_conv_bn(gm: fx.GraphModule, ph_dict: Dict[str, Placeholder], node: fx.Node):
    """
    Constant folding the batch norm into the conv's weights and bias.

    Before:
        x --> [conv_old] --> x --> [bn]

    After:
        x --> [conv_new]
    """

    if node.op == "call_function" and node.target == F.batch_norm:
        bn_node = node
        bn_input, mean_node, var_node, bnW_node, bnBias_node, _, _, eps = bn_node.args

        if not (
            isinstance(bn_input, fx.Node)
            and isinstance(mean_node, fx.Node)
            and isinstance(var_node, fx.Node)
            and isinstance(bnW_node, fx.Node)
            and isinstance(bnBias_node, fx.Node)
            and isinstance(eps, float)
        ):
            return

        if bn_input.op == "call_function" and bn_input.target == F.conv2d:
            conv_node = bn_input
            _, convW_node, convBias_node, *_ = conv_node.args # assuming bias is disabled for resnet 

            if not isinstance(convW_node, fx.Node):
                return

            convW_old = ph_dict.get(str(convW_node.target), None)
            bnW = ph_dict.get(str(bnW_node.target), None)
            bnBias = ph_dict.get(str(bnBias_node.target), None)
            mean = ph_dict.get(str(mean_node.target), None)
            var = ph_dict.get(str(var_node.target), None)

            if not (
                isinstance(convW_old, Placeholder)
                and isinstance(bnW, Placeholder)
                and isinstance(bnBias, Placeholder)
                and isinstance(mean, Placeholder)
                and isinstance(var, Placeholder)
            ):
                return

            if var is None or mean is None:
                return

            # hardcoded no convBias case for resnet:

            inv_sqrt_var_eps = (var.value + eps) ** -0.5
            convW_new = convW_old.value * (bnW.value * inv_sqrt_var_eps).view(
                -1, 1, 1, 1
            )

            ph_dict[str(convW_node.target)].value.data.copy_(convW_new)

            convBias_old = torch.zeros(
                convW_old.value.size(0),  # out channels
                dtype=convW_old.value.dtype,
                device=convW_old.value.device,
            )

            conv_node.args = (
                conv_node.args[0],
                conv_node.args[1],
                bnBias_node,
                *conv_node.args[3:],
            )

            convBias_new = (
                convBias_old - mean.value
            ) * bnW.value * inv_sqrt_var_eps + bnBias.value

            # rename
            convBias_node_target = str(convW_node.target).replace("weight", "bias")
            ph_dict[convBias_node_target] = ph_dict[str(bnBias_node.target)]
            bnBias_node.target = convBias_node_target
            bnBias_node.name = str(convW_node.name).replace("weight", "bias")

            ph_dict[convBias_node_target].value.data.copy_(convBias_new)

            # remove batch norm from graph
            bn_node.replace_all_uses_with(conv_node)
            gm.graph.erase_node(bn_node)

            gm.graph.erase_node(mean_node)
            gm.graph.erase_node(var_node)
            gm.graph.erase_node(bnW_node)

            del ph_dict[str(mean_node.target)]
            del ph_dict[str(var_node.target)]
            del ph_dict[str(bnW_node.target)]


@triton.jit
def add_relu_kernel(lhs, rhs, out, n_elements, BLOCKDIM: tl.constexpr):
    blockids = tl.program_id(0)  # x dim
    tids = blockids * BLOCKDIM + tl.arange(0, BLOCKDIM)

    mask = tids < n_elements

    x = tl.load(lhs + tids, mask)
    y = tl.load(rhs + tids, mask)

    x = tl.maximum(0, x + y)  # add + relu
    tl.store(out + tids, x, mask)


BLOCK_DIM = 256


@torch.library.custom_op("visionrt::add_relu", mutates_args=())
def add_relu(conv_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(conv_out)
    n_elements = conv_out.numel()

    grid = ((n_elements + BLOCK_DIM - 1) // BLOCK_DIM,)
    compiled_kernel = add_relu_kernel[grid](conv_out, residual, out, n_elements, BLOCKDIM=BLOCK_DIM)  # type: ignore[arg-type]

    if config.debug:
        pass
    # if isinstance(compiled_kernel, object):
    # print(compiled_kernel.asm["llir"])  # prinitng llvm ir to see vectorization

    return out


#@torch.library.custom_op("visionrt::add_relu_cuda", mutates_args=())
#def add_relu_cuda(conv_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
#    return fused_add_relu_cuda(conv_out, residual)


@add_relu.register_fake
def _(conv_out, residual):
    return F.relu(conv_out + residual)


#@add_relu_cuda.register_fake
#def _(conv_out, residual):
#    return torch.relu(conv_out + residual)


make_fallback(torch.ops.visionrt.add_relu)
#make_fallback(torch.ops.visionrt.add_relu_cuda)


@register_transformation("fuse_add_relu")
def fuse_add_relu(gm: fx.GraphModule, ph_dict: Dict[str, Placeholder], node: fx.Node):
    """
    Kernel fusing residual add and relu.

    ```
        Before:
            residual --\
                        [add] --> [relu]
            conv_out --/

        After:
            residual --\
                        [add_relu]
            conv_out --/
    ```
    """

    if node.op == "call_function" and node.target == F.relu:
        relu_node = node

        relu_input, *_ = relu_node.args

        if not isinstance(relu_input, fx.Node):
            return

        if relu_input.op == "call_function" and relu_input.target == operator.iadd:
            iadd_node = relu_input
            # lhs, rhs = iadd_node.args | lhs is conv node, rhs is residual node

            # replace iadd fn with fused add-relu fn
            with gm.graph.inserting_before(relu_node):
                fused_node = gm.graph.call_function(
                    torch.ops.visionrt.add_relu,  # use triton kernel for different precisions
                    args=iadd_node.args,
                    name=iadd_node.name,
                )

            # remove relu
            relu_node.replace_all_uses_with(fused_node)
            gm.graph.erase_node(relu_node)
            gm.graph.erase_node(iadd_node)


def optimize_fx(
    gm: nn.Module,
    placeholders: Dict[str, Placeholder],
    transformations: List[Transformation],
):
    if not isinstance(gm, fx.GraphModule):
        gm = fx.symbolic_trace(gm)

    for node in list(gm.graph.nodes):

        # TODO: downsample second stream

        for xform in transformations:
            xform(gm, placeholders, node)

    inputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            inputs.append(placeholders[str(node.target)].value)

    gm.recompile()
    gm.graph.lint()
    return gm, inputs


def inspect_fx(gm, inputs, return_input_indices=False):
    line = "=" * 50

    if config.debug:
        print(line)
        print("FX GRAPH: POST CUSTOM BACKEND")
        print(line)

    input_idx = 0
    indicies = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            assert input_idx < len(inputs)

            input_type = "INPUT"
            x = inputs[input_idx]

            if isinstance(x, nn.Parameter):
                input_type = "PARAMETER"

            elif isinstance(x, nn.Buffer):
                input_type = "BUFFER"

            if return_input_indices and input_type == "INPUT":
                indicies.append(input_idx)

            if config.debug:
                print(node.name, x.size(), input_type)

            input_idx += 1
        else:
            if config.debug:
                print(node.op, node.name, node.target)

    if config.debug:
        print(line)

    return indicies if return_input_indices else [0]
