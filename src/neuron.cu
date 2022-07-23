
#include "neuron.h"

__global__ void update_dendrites(dendrite* den, int N) {
    // thread info
    int id = threadIdx.x;
    if (id >= N) return;

    // get the working dendrite
    dendrite *d = den + id;

    // discout the current dendrite values
    d->Y *= d->gamma;
    d->approx_plus *= d->gamma_plus;
    d->approx_minus *= d->gamma_minus;

    // add the input values to the transfer function
    float X = *(den->X);
    den->Y += X * d->k;
    den->approx_plus += X * d->approx_multi;
    den->approx_minus += X * d->approx_multi;
}

