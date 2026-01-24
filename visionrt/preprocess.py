import numpy
import torch

import triton
import triton.language as tl


Y_OFFSET = tl.constexpr(16)
UV_OFFSET = tl.constexpr(128)
YUV2RGB_11 = tl.constexpr(298)
YUV2RGB_12 = tl.constexpr(-1)
YUV2RGB_13 = tl.constexpr(409)
YUV2RGB_22 = tl.constexpr(-100)
YUV2RGB_23 = tl.constexpr(-210)
YUV2RGB_32 = tl.constexpr(519)
YUV2RGB_33 = tl.constexpr(0)


@triton.jit
def _normalized_yuyv2rgb_kernel(
    yuyv_ptr,
    out_ptr,
    stride,
    length,
    SCALE_R: tl.constexpr,
    SCALE_G: tl.constexpr,
    SCALE_B: tl.constexpr,
    OFFSET_R: tl.constexpr,
    OFFSET_G: tl.constexpr,
    OFFSET_B: tl.constexpr,
    BLOCKDIM: tl.constexpr,
):
    blockIdx = tl.program_id(0)
    threadIdx = tl.arange(0, BLOCKDIM)

    pair_idx = blockIdx * BLOCKDIM + threadIdx
    total_pairs = length >> 2
    mask = pair_idx < total_pairs

    yuyv_pair = tl.load(yuyv_ptr + pair_idx, mask=mask, other=0).to(tl.uint32)

    Y0 = ((yuyv_pair & 0xFF) - Y_OFFSET).to(tl.int32)
    U = (((yuyv_pair >> 8) & 0xFF) - UV_OFFSET).to(tl.int32)
    Y1 = (((yuyv_pair >> 16) & 0xFF) - Y_OFFSET).to(tl.int32)
    V = (((yuyv_pair >> 24) & 0xFF) - UV_OFFSET).to(tl.int32)

    uv_r = YUV2RGB_12 * U + YUV2RGB_13 * V
    uv_g = YUV2RGB_22 * U + YUV2RGB_23 * V
    uv_b = YUV2RGB_32 * U + YUV2RGB_33 * V

    y0_scaled = YUV2RGB_11 * Y0
    y1_scaled = YUV2RGB_11 * Y1

    R0 = tl.maximum(0, tl.minimum(255.0, (y0_scaled + uv_r) >> 8))
    G0 = tl.maximum(0, tl.minimum(255.0, (y0_scaled + uv_g) >> 8))
    B0 = tl.maximum(0, tl.minimum(255.0, (y0_scaled + uv_b) >> 8))

    R1 = tl.maximum(0, tl.minimum(255.0, (y1_scaled + uv_r) >> 8))
    G1 = tl.maximum(0, tl.minimum(255.0, (y1_scaled + uv_g) >> 8))
    B1 = tl.maximum(0, tl.minimum(255.0, (y1_scaled + uv_b) >> 8))

    R0_n = R0 * SCALE_R + OFFSET_R
    R1_n = R1 * SCALE_R + OFFSET_R

    G0_n = G0 * SCALE_G + OFFSET_G
    G1_n = G1 * SCALE_G + OFFSET_G

    B0_n = B0 * SCALE_B + OFFSET_B
    B1_n = B1 * SCALE_B + OFFSET_B

    pixel_base = pair_idx << 1
    R0_index = pixel_base
    R1_index = pixel_base + 1

    G0_index = R0_index + stride
    G1_index = R1_index + stride

    B0_index = G0_index + stride
    B1_index = G1_index + stride

    tl.store(out_ptr + R0_index, R0_n, mask=mask)
    tl.store(out_ptr + R1_index, R1_n, mask=mask)

    tl.store(out_ptr + G0_index, G0_n, mask=mask)
    tl.store(out_ptr + G1_index, G1_n, mask=mask)

    tl.store(out_ptr + B0_index, B0_n, mask=mask)
    tl.store(out_ptr + B1_index, B1_n, mask=mask)


def _preprocess(
    frame: torch.Tensor,
    stride: int,
    scale: list,
    offset: list,
):
    out = torch.empty(3 * stride, dtype=torch.float32, device=frame.device)
    num_pairs = frame.numel()
    length = num_pairs * 4

    grid = ((num_pairs + 256 - 1) // 256,)

    _normalized_yuyv2rgb_kernel[grid](
        yuyv_ptr=frame.flatten(),
        out_ptr=out,
        stride=stride,
        length=length,
        SCALE_R=scale[0],
        SCALE_G=scale[1],
        SCALE_B=scale[2],
        OFFSET_R=offset[0],
        OFFSET_G=offset[1],
        OFFSET_B=offset[2],
        BLOCKDIM=tl.constexpr(256),
    )

    return out


class Preprocessor:
    def __init__(
        self, mean: list = [0.485, 0.456, 0.406], stdev: list = [0.229, 0.224, 0.225]
    ) -> None:

        assert len(mean) == 3
        assert len(stdev) == 3

        self._scale = [1 / (255 * stdev[i]) for i in range(3)]
        self._offset = [-mean[i] / stdev[i] for i in range(3)]

    def __call__(self, frame: numpy.ndarray):
        height, width, *_ = frame.shape
        frame_flat = frame.reshape(-1).view(numpy.uint32)
        frame_tensor = torch.from_numpy(frame_flat).to(device="cuda")

        rgb = _preprocess(
            frame=frame_tensor,
            stride=height * width,
            scale=self._scale,
            offset=self._offset,
        )

        return rgb.reshape(1, 3, height, width)
