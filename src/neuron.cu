
#include "neuron.h"

__global__ void update_dendrites_cuda(Dendrite* den, float X, int N) {
    // thread info
    int id = threadIdx.x;
    if (id >= N) return;

    // get the working dendrite
    Dendrite *d = &den[id];

    // discout the current dendrite values
    d->Y *= d->gamma;
    d->approx_plus *= d->gamma_plus;
    d->approx_minus *= d->gamma_minus;

    // add the input values to the transfer function
    den->Y += X * d->k;
    den->approx_plus += X * d->approx_multi;
    den->approx_minus += X * d->approx_multi;
}

void update_dendrites(Dendrite* den, float X, int N) {
    update_dendrites_cuda<<<1, 1>>>(den, X, N);
}