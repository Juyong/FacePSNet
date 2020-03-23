#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>
#include "cuda_config.h"

void gen_normal_mask_cuda(float *proj_geo, float *tri_normal, long *tri_inds, float *depthBuffer,
                          int *dBufferLocked, float *normal_map, float *mask, int batch_size,
                          int point_num, int tri_num, int height, int width);

void gen_depth_cuda(float *proj_geo, long *tri_inds, float *depthBuffer,
                    int *dBufferLocked, int batch_size,
                    int point_num, int tri_num, int height, int width);

// C++ interface

#define CHECK_CUDA(x) \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

std::vector<at::Tensor> gen_normal_mask(at::Tensor proj_geo, at::Tensor tri_normal,
                                        at::Tensor tri_inds, int height, int width)
{
    CHECK_INPUT(proj_geo);
    CHECK_INPUT(tri_normal);
    CHECK_INPUT(tri_inds);

    int batch_size = proj_geo.size(0);
    int point_num = proj_geo.size(1);
    int tri_num = tri_inds.size(0);

    auto options_int_nograd = torch::TensorOptions()
                                  .dtype(torch::kInt32)
                                  .layout(proj_geo.layout())
                                  .device(proj_geo.device())
                                  .requires_grad(false);
    auto options_float_nograd = torch::TensorOptions()
                                    .dtype(proj_geo.dtype())
                                    .layout(proj_geo.layout())
                                    .device(proj_geo.device())
                                    .requires_grad(false);

    auto depthBuffer =
        -100000 * torch::ones({batch_size, height, width}, options_float_nograd);
    auto dBufferLocked =
        torch::zeros({batch_size, height, width}, options_int_nograd);
    auto normal_map = torch::zeros({batch_size, 3, height, width}, options_float_nograd);
    auto mask = torch::zeros({batch_size, 1, height, width}, options_float_nograd);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    gen_normal_mask_cuda(proj_geo.data<float>(), tri_normal.data<float>(), tri_inds.data<long>(),
                         depthBuffer.data<float>(), dBufferLocked.data<int>(), normal_map.data<float>(), mask.data<float>(),
                         batch_size, point_num, tri_num, height, width);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    return {normal_map, mask};
}

std::vector<at::Tensor> gen_depth(at::Tensor proj_geo, at::Tensor tri_inds, int height, int width)
{
    CHECK_INPUT(proj_geo);
    CHECK_INPUT(tri_inds);

    int batch_size = proj_geo.size(0);
    int point_num = proj_geo.size(1);
    int tri_num = tri_inds.size(0);

    auto options_int_nograd = torch::TensorOptions()
                                  .dtype(torch::kInt32)
                                  .layout(proj_geo.layout())
                                  .device(proj_geo.device())
                                  .requires_grad(false);
    auto options_float_nograd = torch::TensorOptions()
                                    .dtype(proj_geo.dtype())
                                    .layout(proj_geo.layout())
                                    .device(proj_geo.device())
                                    .requires_grad(false);

    auto depthBuffer =
        -100000 * torch::ones({batch_size, height, width}, options_float_nograd);
    auto dBufferLocked =
        torch::zeros({batch_size, height, width}, options_int_nograd);

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    gen_depth_cuda(proj_geo.data<float>(), tri_inds.data<long>(),
                   depthBuffer.data<float>(), dBufferLocked.data<int>(),
                   batch_size, point_num, tri_num, height, width);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    return {depthBuffer};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gen_normal_mask", &gen_normal_mask, "Gen Normal Mask (CUDA)");
    m.def("gen_depth", &gen_depth, "Gen Depth (CUDA)");
}