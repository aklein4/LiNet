
#include "cuda.h"

#include <chrono>

int time_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

__global__ void cuda_vecs(float* A, float* B, float* C, int N) {
    int id = threadIdx.x;

    if (id < N) {
        C[id] = A[id] + B[id];
    }

}

int add_vecs(float* a, float* b, float* c, int N, int R) {

    int bytes = N * sizeof(float);

    // gpu vectors
    float* d_a;
    float* d_b;
    float* d_c;

    // allocate gpu memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy memory to gpu
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    // create make block size and count
    int block_size = 320;
    int block_count = N/block_size;

    // call the kernel
    int profile = time_ms();
    for (int i = 0; i < R; i++) {
        cuda_vecs<<<block_size, block_count>>>(d_a, d_b, d_c, N);
    }
    profile = time_ms() - profile;

    // get the results back
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    // free the gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return profile;
}