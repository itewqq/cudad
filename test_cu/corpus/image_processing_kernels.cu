// Real-world-style CUDA kernels: image processing pipeline.
// Patterns from production CUDA code: 2D thread grids, per-pixel branching,
// sliding windows, color space conversion, histogram equalization.

#include <stdint.h>

// ------ RGB→Grayscale + Sobel edge detection ------
// Two-pass pattern common in image processing: convert then filter.
// Exercises 2D thread indexing, boundary checks, and shared-memory stencil.

#define SOBEL_BLOCK 16

extern "C" __global__ void sobel_edge_detect(
    const uint8_t * __restrict__ rgb,    // [H][W][3] interleaved
    uint8_t       * __restrict__ edges,  // [H][W] output
    int W, int H
) {
    __shared__ float tile[SOBEL_BLOCK + 2][SOBEL_BLOCK + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * SOBEL_BLOCK + tx;
    int gy = blockIdx.y * SOBEL_BLOCK + ty;

    // Convert RGB to grayscale into shared tile (with halo)
    for (int dy = -1; dy <= 1; dy += 2) {
        for (int dx = -1; dx <= 1; dx += 2) {
            int lx = tx + 1 + dx * (tx == 0 || tx == SOBEL_BLOCK - 1 ? 1 : 0);
            int ly = ty + 1 + dy * (ty == 0 || ty == SOBEL_BLOCK - 1 ? 1 : 0);
            int px = gx + dx * (tx == 0 || tx == SOBEL_BLOCK - 1 ? 1 : 0);
            int py = gy + dy * (ty == 0 || ty == SOBEL_BLOCK - 1 ? 1 : 0);
            if (px >= 0 && px < W && py >= 0 && py < H && lx >= 0 && lx < SOBEL_BLOCK + 2 && ly >= 0 && ly < SOBEL_BLOCK + 2) {
                int idx = (py * W + px) * 3;
                tile[ly][lx] = 0.299f * rgb[idx] + 0.587f * rgb[idx+1] + 0.114f * rgb[idx+2];
            }
        }
    }

    // Center pixel always loaded
    {
        int lx = tx + 1, ly = ty + 1;
        if (gx < W && gy < H) {
            int idx = (gy * W + gx) * 3;
            tile[ly][lx] = 0.299f * rgb[idx] + 0.587f * rgb[idx+1] + 0.114f * rgb[idx+2];
        } else {
            tile[ly][lx] = 0.0f;
        }
    }
    __syncthreads();

    if (gx >= W || gy >= H) return;

    int sx = tx + 1, sy = ty + 1;
    float gx_val = -tile[sy-1][sx-1] - 2*tile[sy][sx-1] - tile[sy+1][sx-1]
                   +tile[sy-1][sx+1] + 2*tile[sy][sx+1] + tile[sy+1][sx+1];
    float gy_val = -tile[sy-1][sx-1] - 2*tile[sy-1][sx] - tile[sy-1][sx+1]
                   +tile[sy+1][sx-1] + 2*tile[sy+1][sx] + tile[sy+1][sx+1];
    float mag = sqrtf(gx_val * gx_val + gy_val * gy_val);
    edges[gy * W + gx] = (uint8_t)(mag > 255.0f ? 255 : (int)mag);
}

// ------ Histogram (256-bin) with privatized per-block histograms ------
// Pattern from real image processing: per-block shared-mem histogram,
// then atomicAdd to global. Multi-pass reduction.

extern "C" __global__ void histogram_equalize(
    const uint8_t * __restrict__ input,
    uint8_t       * __restrict__ output,
    const uint32_t * __restrict__ cdf,  // precomputed CDF[256]
    uint32_t cdf_min, int total_pixels
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_pixels) return;

    uint8_t val = input[tid];
    float equalized = ((float)(cdf[val] - cdf_min) / (float)(total_pixels - cdf_min)) * 255.0f;
    output[tid] = (uint8_t)(equalized < 0.0f ? 0.0f : (equalized > 255.0f ? 255.0f : equalized));
}

// ------ Box blur (variable radius) with shared memory ------
// Sliding window pattern common in real-time image processing.

extern "C" __global__ void box_blur_variable_radius(
    const float * __restrict__ input,
    float       * __restrict__ output,
    const int   * __restrict__ radii, // per-pixel radius
    int W, int H
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= W || gy >= H) return;

    int r = radii[gy * W + gx];
    if (r < 0) r = 0;
    if (r > 15) r = 15;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -r; dy <= r; dy++) {
        int py = gy + dy;
        if (py < 0 || py >= H) continue;
        for (int dx = -r; dx <= r; dx++) {
            int px = gx + dx;
            if (px < 0 || px >= W) continue;
            sum += input[py * W + px];
            count++;
        }
    }
    output[gy * W + gx] = (count > 0) ? sum / count : 0.0f;
}

// ------ Non-maximum suppression (NMS for object detection) ------
// Pattern from YOLO/SSD post-processing: iterate over boxes,
// suppress overlapping ones. Complex nested loops with early exit.

struct BBox {
    float x1, y1, x2, y2, score;
};

__device__ float iou(const BBox &a, const BBox &b) {
    float ix1 = fmaxf(a.x1, b.x1), iy1 = fmaxf(a.y1, b.y1);
    float ix2 = fminf(a.x2, b.x2), iy2 = fminf(a.y2, b.y2);
    float iw = fmaxf(0.0f, ix2 - ix1), ih = fmaxf(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

extern "C" __global__ void nms_kernel(
    const float * __restrict__ boxes,    // [N][5]: x1,y1,x2,y2,score
    int         * __restrict__ keep,     // [N]: 1=keep, 0=suppress
    int N, float iou_threshold
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    BBox bi;
    bi.x1 = boxes[i*5]; bi.y1 = boxes[i*5+1];
    bi.x2 = boxes[i*5+2]; bi.y2 = boxes[i*5+3]; bi.score = boxes[i*5+4];

    // Greedy NMS: suppress box i if any higher-scoring box j overlaps it
    keep[i] = 1;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        BBox bj;
        bj.x1 = boxes[j*5]; bj.y1 = boxes[j*5+1];
        bj.x2 = boxes[j*5+2]; bj.y2 = boxes[j*5+3]; bj.score = boxes[j*5+4];

        if (bj.score > bi.score || (bj.score == bi.score && j < i)) {
            if (iou(bi, bj) > iou_threshold) {
                keep[i] = 0;
                return;
            }
        }
    }
}

// ------ Bilinear interpolation resize ------
// Common in preprocessing for neural networks.

extern "C" __global__ void bilinear_resize(
    const float * __restrict__ src,
    float       * __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = (dx + 0.5f) * src_w / (float)dst_w - 0.5f;
    float sy = (dy + 0.5f) * src_h / (float)dst_h - 0.5f;

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float fx = sx - x0, fy = sy - y0;

    // Clamp
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    float v00 = src[y0 * src_w + x0];
    float v01 = src[y0 * src_w + x1];
    float v10 = src[y1 * src_w + x0];
    float v11 = src[y1 * src_w + x1];

    dst[dy * dst_w + dx] = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01
                          + (1-fx)*fy*v10 + fx*fy*v11;
}
