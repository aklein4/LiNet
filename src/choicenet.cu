
#include "choicenet.h"
#include <iostream>
#include <assert.h>

ChoiceNet::ChoiceNet() {
    Synapse* d = new Synapse();

    cudaError_t rval = cudaMalloc(&gpu_d, sizeof(Synapse));
    assert(rval == cudaSuccess);
    rval = cudaMemcpy(gpu_d, d, sizeof(Synapse), cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
    delete d;
}

ChoiceNet::~ChoiceNet() {
    cudaFree(gpu_d);
}

float ChoiceNet::update(float x_in) {
    //update_Synapses(gpu_d, x_in, 1);
    Synapse* d_out = new Synapse();
    cudaError_t rval = cudaMemcpy(d_out, gpu_d, sizeof(Synapse), cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    float Y = d_out->Y;
    delete d_out;
    return Y;
}