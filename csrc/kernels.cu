#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "kernels.cuh"

// yuyv2rgb Credits: https://stackoverflow.com/questions/72056909/convert-yuv2-yuyv-frames-to-rgb-without-use-of-opencv

constexpr int Y_OFFSET = 16;
constexpr int UV_OFFSET = 128;
constexpr int YUV2RGB_11 = 298;
constexpr int YUV2RGB_12 = -1;
constexpr int YUV2RGB_13 = 409;
constexpr int YUV2RGB_22 = -100;
constexpr int YUV2RGB_23 = -210;
constexpr int YUV2RGB_32 = 519;
constexpr int YUV2RGB_33 = 0;

#define clamp(x) (max(0, min(255, (x))))

__global__ void yuyv2rgb_chw_kernel(
    const uint32_t* __restrict__ yuyv,
    float* __restrict__ rgb,
    int num_pairs,
    int stride,
    float scale_r, float scale_g, float scale_b,
    float offset_r, float offset_g, float offset_b
) {
    const int pair_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (pair_idx >= num_pairs) {
        return;
    }

    const uint32_t pair = yuyv[pair_idx];
    const int Y0 = (pair & 0xFF) - Y_OFFSET;
    const int U = ((pair >> 8) & 0xFF) - UV_OFFSET;
    const int Y1 = ((pair >> 16) & 0xFF) - Y_OFFSET;
    const int V = ((pair >> 24) & 0xFF) - UV_OFFSET;

    const int uv_r = YUV2RGB_12 * U + YUV2RGB_13 * V;
    const int uv_g = YUV2RGB_22 * U + YUV2RGB_23 * V;
    const int uv_b = YUV2RGB_32 * U + YUV2RGB_33 * V;

    const int y0_scaled = YUV2RGB_11 * Y0;
    const int R0 = clamp((y0_scaled + uv_r) >> 8);
    const int G0 = clamp((y0_scaled + uv_g) >> 8);
    const int B0 = clamp((y0_scaled + uv_b) >> 8);

    const int y1_scaled = YUV2RGB_11 * Y1;
    const int R1 = clamp((y1_scaled + uv_r) >> 8);
    const int G1 = clamp((y1_scaled + uv_g) >> 8);
    const int B1 = clamp((y1_scaled + uv_b) >> 8);

    const float R0_n = R0 * scale_r + offset_r;
    const float R1_n = R1 * scale_r + offset_r;
    const float G0_n = G0 * scale_g + offset_g;
    const float G1_n = G1 * scale_g + offset_g;
    const float B0_n = B0 * scale_b + offset_b;
    const float B1_n = B1 * scale_b + offset_b;

    const int pixel_base = pair_idx << 1;

    rgb[pixel_base] = R0_n;
    rgb[pixel_base + 1] = R1_n;
    rgb[stride + pixel_base] = G0_n;
    rgb[stride + pixel_base + 1] = G1_n;
    rgb[2 * stride + pixel_base] = B0_n;
    rgb[2 * stride + pixel_base + 1] = B1_n;
}

torch::Tensor launch_yuyv2rgb_chw(
    const torch::Tensor& yuyv,  // [H, W, 2] as uint8
    int height,
    int width,
    const std::vector<float>& scale,
    const std::vector<float>& offset
) {
    const int num_pairs = (height * width) / 2;  // Number of YUYV pairs
    const int stride = height * width;

    auto out = torch::empty({3 * stride}, torch::TensorOptions().dtype(torch::kFloat32).device(yuyv.device()));

    constexpr int block_dim = 256;
    const int n_blocks = (num_pairs + block_dim - 1) / block_dim;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    yuyv2rgb_chw_kernel<<<n_blocks, block_dim, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(yuyv.data_ptr()),
        out.data_ptr<float>(),
        num_pairs,
        stride,
        scale[0], scale[1], scale[2],
        offset[0], offset[1], offset[2]
    );

    return out;
}


__global__ void add_relu( // forcing vectorized float4
    const float4* a, // [bs, in_ch, h, w]
    const float4* b, // [bs, in_ch, h, w]
    float4* c, // [bs, in_ch, h, w]
    const int n_vecs // bs * in_ch * h * w / 4
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_vecs) {
        return;
    }

    float4 a_vec = a[i];
    float4 b_vec = b[i];
    float4 c_vec;

    c_vec.x = fmax(a_vec.x + b_vec.x, 0.0f);
    c_vec.y = fmax(a_vec.y + b_vec.y, 0.0f);
    c_vec.z = fmax(a_vec.z + b_vec.z, 0.0f);
    c_vec.w = fmax(a_vec.w + b_vec.w, 0.0f);

    c[i] = c_vec;
}

torch::Tensor launch_add_relu(
    const torch::Tensor& lhs, // [bs, ch, h, w]
    const torch::Tensor& rhs // [bs, ch, h, w]
) {

    const int bs = lhs.size(0);
    const int ch = lhs.size(1);
    const int h = lhs.size(2);
    const int w = lhs.size(3);

    const int n_vecs = (bs * ch * h * w) / 4;
    auto out = torch::empty_like(lhs);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    add_relu<<<(n_vecs + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs.data_ptr()),
        reinterpret_cast<const float4*>(rhs.data_ptr()),
        reinterpret_cast<float4*>(out.data_ptr()),
        n_vecs
    );

    return out;
}

