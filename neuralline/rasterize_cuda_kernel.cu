#include <ATen/ATen.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// When set to 1024, the backward kernel fails :(
// Referring to caffe, it is 512.
#ifndef MAX_THREADS
#define MAX_THREADS 512
#endif


// For the older gpus atomicAdd with double arguments does not exist
// https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/cuda/rasterize_cuda_kernel.cu
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


// https://stackoverflow.com/a/14038590
#define GPU_ERROR_CHECK(ans) {gpu_assert((ans), __FILE__, __LINE__);}
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"\nGPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}


inline int gpu_blocks(int total_threads, int threads_per_block) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}


namespace { // kernel namespace

/* Helper functions */
template <typename scalar_t=float>
__device__ __forceinline__ scalar_t vec2f_squared_norm(const scalar_t* v) {
    return v[0] * v[0] + v[1] * v[1];
}


template <typename scalar_t=float>
__global__ void rasterize_cuda_forward_kernel(
    // Outputs
    scalar_t*       __restrict__ line_map,
    int32_t*        __restrict__ line_index_map,
    scalar_t*       __restrict__ line_weight_map,
    // Temps
    int32_t*        __restrict__ locks,
    int                          num_intensity_channels,
    int                          num_lines,
    int                          loops,
    // Inputs
    const scalar_t* __restrict__ lines,
    const scalar_t* __restrict__ intensities,
    int                          img_size,
    scalar_t                     thickness,
    scalar_t                     eps) {
    
    // lines: [batch_size, num_lines, 3]
    // intensities: [batch_size, num_lines, channels]
    
    // line_map: [batch_size, num_intensity_channels, img_size, img_size]
    // line_index_map: [batch_size, 1, img_size, img_size]
    // line_weight_map: [batch_size, 1, img_size, img_size]
    // locks: [batch_size, 1, img_size, img_size]
    
    // Global line id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const int batchId = i / num_lines;
    const int lineId  = i % num_lines;
    // (x, y, pen_state), range: [-1.0, 1.0]
    const scalar_t* line = &lines[i * 3];
    if (line[2] > 0 || lineId == (num_lines - 1))
        // Skip this line if p_i's pen_state == 1
        return;
    
    // Convert (x, y) from [-1.0, 1.0] to [0, img_size-1]
    scalar_t tl[2][2]; 
    for (int vidx = 0; vidx < 2; ++vidx)
        for (int axis = 0; axis < 2; ++axis) // Consider p_i and p_i+1
            tl[vidx][axis] = (line[vidx * 3 + axis] + 1.0f) * (img_size - 1.0f) / 2.0f;
    
    // Direction vector from p_i to p_i+1
    scalar_t tlVec[2]      = {tl[1][0] - tl[0][0], tl[1][1] - tl[0][1]};
    const scalar_t length2 = vec2f_squared_norm<scalar_t>(tlVec);
    const scalar_t length  = sqrt(length2);
    // If p_i and p_i+1 are too close, skip this line
    if (length < eps) return;
    
    // The bounding box of the line segment
    int xiMin = max(floor(min(tl[0][0], tl[1][0]) - thickness), 0.0f);
    int xiMax = min(ceil(max(tl[0][0], tl[1][0]) + thickness), img_size - 1.0f);
    int yiMin = max(floor(min(tl[0][1], tl[1][1]) - thickness), 0.0f);
    int yiMax = min(ceil(max(tl[0][1], tl[1][1]) + thickness), img_size - 1.0f);
    
    const int imgSize2 = img_size * img_size;

    // Rasterization: test each pixel in the bounding box
    for (int xi = xiMin; xi <= xiMax; ++xi) {
        for (int yi = yiMin; yi <= yiMax; ++yi) {
            // Compute distance from a point (xi, yi) to the line segment
            scalar_t pv0Vec[2] = {xi - tl[0][0], yi - tl[0][1]};
            // ratio range: [0, 1]
            scalar_t ratio     = max(min((pv0Vec[0] * tlVec[0] + pv0Vec[1] * tlVec[1]) / length2, 1.0f), 0.0f);
            scalar_t pProj[2]  = {tl[0][0] + ratio * tlVec[0], tl[0][1] + ratio * tlVec[1]};
            scalar_t ppProj[2] = {xi - pProj[0], yi - pProj[1]};
            scalar_t dist      = sqrt(vec2f_squared_norm<scalar_t>(ppProj));

            // Too far away from the line segment, skip this pixel
            if (dist > thickness) continue;
            
            int lockId = imgSize2 * batchId + img_size * yi + xi;
            int locked = 0;
            do {
                if ((locked = atomicCAS(&locks[lockId], 0, 1)) == 0) {
                    // Visibility check by point ordering
                    if (atomicAdd(&line_index_map[lockId], 0) < lineId) {
                        atomicExch(&line_index_map[lockId], lineId);
                        atomicExch(&line_weight_map[lockId], ratio);
                        // Linear interpolation
                        const scalar_t* intensity = &intensities[i * num_intensity_channels];
                        for (int cid = 0; cid < num_intensity_channels; ++cid) {
                            scalar_t lerp_val = intensity[cid] + ratio * (intensity[num_intensity_channels + cid] - intensity[cid]);
                            int pixId = num_intensity_channels * imgSize2 * batchId + imgSize2 * cid + img_size * yi + xi;
                            atomicExch(&line_map[pixId], lerp_val);
                        }
                    }
                    atomicExch(&locks[lockId], 0);
                }
            } while (locked > 0);
        }
    }
}


template <typename scalar_t=float>
__global__ void rasterize_cuda_backward_kernel_intensities(
    // Outputs
    scalar_t*       __restrict__   grad_intensities,
    // Temps
    const int32_t*  __restrict__   line_index_map,
    const scalar_t* __restrict__   line_weight_map,
    int                            num_intensity_channels,
    int                            num_lines,
    int                            batch_size,
    int                            loops,
    // Inputs
    const scalar_t* __restrict__   grad_line_map,
    int                            img_size) {
    
    // grad_intensities: [batch_size, num_lines, channels]

    // line_index_map: [batch_size, 1, img_size, img_size]
    // line_weight_map: [batch_size, 1, img_size, img_size]
    // grad_line_map: [batch_size, num_intensity_channels, img_size, img_size]
    
    // Global pixel id (without considering intensity_channels)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= loops) return;

    const int localLineId = line_index_map[i];
    if (localLineId < 0)
        // Current pixel is not covered by any line segments
        return;
    const scalar_t ratio = line_weight_map[i];
    
    const int imgSize2 = img_size * img_size;
    const int batchId = i / imgSize2;
    const int pixId = i % imgSize2;
    const int globalLineId = num_lines * batchId + localLineId;
    
    for (int cid = 0; cid < num_intensity_channels; ++cid) {
        const scalar_t gradLine = grad_line_map[num_intensity_channels * imgSize2 * batchId + imgSize2 * cid + pixId];
        atomicAdd(&grad_intensities[globalLineId * num_intensity_channels + cid], (1 - ratio) * gradLine);
        atomicAdd(&grad_intensities[(globalLineId + 1) * num_intensity_channels + cid], ratio * gradLine);
    }
}

} // namespace


void rasterize_cuda_forward(
    // Forward inputs
    at::Tensor lines,
    at::Tensor intensities,
    // Forward outputs
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    // Temps
    at::Tensor locks,
    int   img_size,
    float thickness,
    float eps) {
    
    // lines: [batch_size, num_lines, 3]
    // intensities: [batch_size, num_lines, channels]
    const auto batch_size = lines.size(0);
    const auto num_lines = lines.size(1);
    const auto num_intensity_channels = intensities.size(2);
 
    const int loops   = batch_size * num_lines;
    const int threads = MAX_THREADS;
    const int blocks  = gpu_blocks(loops, threads);

    rasterize_cuda_forward_kernel<float><<<blocks, threads>>>(
        line_map.data<float>(),
        line_index_map.data<int32_t>(),
        line_weight_map.data<float>(),
        locks.data<int32_t>(),
        num_intensity_channels,
        num_lines,
        loops,
        lines.data<float>(),
        intensities.data<float>(),
        img_size,
        thickness,
        eps);
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}


void rasterize_cuda_backward(
    // Backward outputs
    at::Tensor grad_intensities,
    // Backward inputs
    at::Tensor grad_line_map,
    // Forward outputs
    at::Tensor line_map,
    at::Tensor line_index_map,
    at::Tensor line_weight_map,
    // Forward inputs
    at::Tensor lines,
    at::Tensor intensities,
    int   img_size,
    float thickness,
    float eps) {
    
    // lines: [batch_size, num_lines, 3]
    // intensities: [batch_size, num_lines, channels]
    const auto batch_size = lines.size(0);
    const auto num_lines = lines.size(1);
    const auto num_intensity_channels = intensities.size(2);
    
    const int loops  = batch_size * img_size * img_size;
    const int threads = MAX_THREADS;
    const int blocks = gpu_blocks(loops, threads);

    rasterize_cuda_backward_kernel_intensities<float><<<blocks, threads>>>(
        grad_intensities.data<float>(),
        line_index_map.data<int32_t>(),
        line_weight_map.data<float>(),
        num_intensity_channels,
        num_lines,
        batch_size,
        loops,
        grad_line_map.data<float>(),
        img_size);
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
} 
