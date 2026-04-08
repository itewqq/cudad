// Real-world-style CUDA kernels: physics simulation & graph algorithms.
// Patterns from N-body, SPH fluid, BFS, PageRank — production GPU computing.

#include <stdint.h>

// ------ N-body gravitational force (direct, O(N^2) tile-based) ------
// Classic GPU pattern: each thread accumulates force from all other bodies
// using shared-memory tiling to reduce global memory traffic.

#define NBODY_TILE 256

extern "C" __global__ void nbody_forces(
    const float4 * __restrict__ pos_mass, // [N] {x,y,z,mass}
    float3       * __restrict__ forces,   // [N] {fx,fy,fz}
    int N, float eps2, float G
) {
    __shared__ float4 tile[NBODY_TILE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 my_pos;
    if (tid < N) {
        my_pos = pos_mass[tid];
    } else {
        my_pos = make_float4(0,0,0,0);
    }

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int t = 0; t < (N + NBODY_TILE - 1) / NBODY_TILE; t++) {
        int idx = t * NBODY_TILE + threadIdx.x;
        if (idx < N) {
            tile[threadIdx.x] = pos_mass[idx];
        } else {
            tile[threadIdx.x] = make_float4(0,0,0,0);
        }
        __syncthreads();

        int limit = min(NBODY_TILE, N - t * NBODY_TILE);
        for (int j = 0; j < limit; j++) {
            float dx = tile[j].x - my_pos.x;
            float dy = tile[j].y - my_pos.y;
            float dz = tile[j].z - my_pos.z;
            float dist2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv_dist = rsqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float f = G * my_pos.w * tile[j].w * inv_dist3;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }
        __syncthreads();
    }

    if (tid < N) {
        forces[tid] = make_float3(fx, fy, fz);
    }
}

// ------ Particle-in-cell scatter (PIC plasma sim) ------
// Pattern from plasma physics: each particle deposits charge to
// 4 neighboring grid cells with bilinear weights.

extern "C" __global__ void pic_charge_deposit(
    const float * __restrict__ pos_x,   // [N] particle x
    const float * __restrict__ pos_y,   // [N] particle y
    const float * __restrict__ charges, // [N] particle charge
    float       * __restrict__ grid,    // [GH][GW] charge density
    int N, int GW, int GH, float dx, float dy
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    float px = pos_x[pid], py = pos_y[pid];
    float gx = px / dx, gy = py / dy;

    int ix = (int)floorf(gx), iy = (int)floorf(gy);
    float fx = gx - ix, fy = gy - iy;

    float q = charges[pid];

    // Bilinear deposit to 4 cells
    if (ix >= 0 && ix < GW - 1 && iy >= 0 && iy < GH - 1) {
        atomicAdd(&grid[iy * GW + ix],         q * (1-fx) * (1-fy));
        atomicAdd(&grid[iy * GW + ix + 1],     q * fx * (1-fy));
        atomicAdd(&grid[(iy+1) * GW + ix],     q * (1-fx) * fy);
        atomicAdd(&grid[(iy+1) * GW + ix + 1], q * fx * fy);
    }
}

// ------ BFS frontier expansion (edge-parallel) ------
// Pattern from graph analytics: each thread checks one edge,
// updates distance if target is unvisited.

extern "C" __global__ void bfs_expand(
    const int * __restrict__ row_offsets,  // CSR row pointers [V+1]
    const int * __restrict__ col_indices,  // CSR column indices [E]
    int       * __restrict__ distances,    // [V], -1 = unvisited
    const int * __restrict__ frontier,     // current frontier vertices
    int       * __restrict__ next_frontier,// output frontier
    int       * __restrict__ next_count,   // atomic counter
    int frontier_size, int current_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    int start = row_offsets[u];
    int end = row_offsets[u + 1];

    for (int e = start; e < end; e++) {
        int v = col_indices[e];
        // atomicCAS to claim unvisited vertex
        int old = atomicCAS(&distances[v], -1, current_dist + 1);
        if (old == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = v;
        }
    }
}

// ------ PageRank iteration ------
// Pattern from graph analytics: sparse matrix-vector product with
// damping factor and convergence check.

extern "C" __global__ void pagerank_iter(
    const int   * __restrict__ row_offsets,  // CSR [V+1]
    const int   * __restrict__ col_indices,  // CSR [E]
    const int   * __restrict__ out_degrees,  // [V]
    const float * __restrict__ rank_in,      // [V]
    float       * __restrict__ rank_out,     // [V]
    float       * __restrict__ diff_out,     // [V] abs diff for convergence
    int V, float damping, float base_rank
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;

    float sum = 0.0f;
    int start = row_offsets[v];
    int end = row_offsets[v + 1];

    for (int e = start; e < end; e++) {
        int u = col_indices[e];
        int deg = out_degrees[u];
        if (deg > 0) {
            sum += rank_in[u] / (float)deg;
        }
    }

    float new_rank = base_rank + damping * sum;
    rank_out[v] = new_rank;
    float d = new_rank - rank_in[v];
    diff_out[v] = (d >= 0) ? d : -d;
}

// ------ Lennard-Jones force calculation (molecular dynamics) ------
// Pattern from LAMMPS-style MD: cutoff-based pair interaction with
// neighbor list.

extern "C" __global__ void lj_forces(
    const float * __restrict__ pos,          // [N*3]
    const int   * __restrict__ neighbor_list, // [N * max_neighbors]
    const int   * __restrict__ num_neighbors, // [N]
    float       * __restrict__ forces,        // [N*3]
    int N, int max_neighbors,
    float epsilon, float sigma, float cutoff2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = pos[i*3], yi = pos[i*3+1], zi = pos[i*3+2];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    int nn = num_neighbors[i];
    for (int n = 0; n < nn; n++) {
        int j = neighbor_list[i * max_neighbors + n];
        float dx = xi - pos[j*3];
        float dy = yi - pos[j*3+1];
        float dz = zi - pos[j*3+2];
        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < cutoff2 && r2 > 1e-10f) {
            float r2_inv = 1.0f / r2;
            float s2_r2 = sigma * sigma * r2_inv;
            float s6_r6 = s2_r2 * s2_r2 * s2_r2;
            float force_mag = 24.0f * epsilon * r2_inv * s6_r6 * (2.0f * s6_r6 - 1.0f);
            fx += force_mag * dx;
            fy += force_mag * dy;
            fz += force_mag * dz;
        }
    }

    forces[i*3]   = fx;
    forces[i*3+1] = fy;
    forces[i*3+2] = fz;
}
