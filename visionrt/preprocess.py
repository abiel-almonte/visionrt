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
def _yuyv2rgb_kernel(
    yuyv_ptr,
    out_ptr,
    stride,
    num_pairs,
    SCALE_R: tl.constexpr,
    SCALE_G: tl.constexpr,
    SCALE_B: tl.constexpr,
    OFFSET_R: tl.constexpr,
    OFFSET_G: tl.constexpr,
    OFFSET_B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pairs

    yuyv = tl.load(yuyv_ptr + offsets, mask=mask, other=0).to(tl.uint32)

    Y0 = ((yuyv & 0xFF) - Y_OFFSET).to(tl.int32)
    U = (((yuyv >> 8) & 0xFF) - UV_OFFSET).to(tl.int32)
    Y1 = (((yuyv >> 16) & 0xFF) - Y_OFFSET).to(tl.int32)
    V = (((yuyv >> 24) & 0xFF) - UV_OFFSET).to(tl.int32)

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

    px = offsets * 2

    tl.store(out_ptr + px, R0 * SCALE_R + OFFSET_R, mask=mask)
    tl.store(out_ptr + px + 1, R1 * SCALE_R + OFFSET_R, mask=mask)

    tl.store(out_ptr + stride + px, G0 * SCALE_G + OFFSET_G, mask=mask)
    tl.store(out_ptr + stride + px + 1, G1 * SCALE_G + OFFSET_G, mask=mask)

    tl.store(out_ptr + 2 * stride + px, B0 * SCALE_B + OFFSET_B, mask=mask)
    tl.store(out_ptr + 2 * stride + px + 1, B1 * SCALE_B + OFFSET_B, mask=mask)


BLOCK_SIZE = tl.constexpr(256)


class Preprocessor:
    def __init__(
        self,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
    ) -> None:
        assert len(mean) == 3 and len(std) == 3
        self._scale = tuple(1 / (255 * std[i]) for i in range(3))
        self._offset = tuple(-mean[i] / std[i] for i in range(3))

    def __call__(self, frame: numpy.ndarray) -> torch.Tensor:
        h, w = frame.shape[:2]
        num_pairs = (h * w) // 2
        stride = h * w

        yuyv = torch.from_numpy(frame.ravel().view(numpy.uint32)).cuda(non_blocking=True)
        out = torch.empty(3 * stride, dtype=torch.float32, device="cuda")

        grid = ((num_pairs + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _yuyv2rgb_kernel[grid](
            yuyv, out, stride, num_pairs,
            *self._scale, *self._offset,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out.view(1, 3, h, w)
