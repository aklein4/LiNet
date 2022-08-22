
#include "choicenet.h"
#include <iostream>
#include <assert.h>

ChoiceNet::ChoiceNet() {
    Dendrite* d = new Dendrite();

    cudaError_t rval = cudaMalloc(&gpu_d, sizeof(Dendrite));
    assert(rval == cudaSuccess);
    rval = cudaMemcpy(gpu_d, d, sizeof(Dendrite), cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
    delete d;
}

ChoiceNet::~ChoiceNet() {
    cudaFree(gpu_d);
}

float ChoiceNet::update(float x_in) {
    //update_Dendrites(gpu_d, x_in, 1);
    Dendrite* d_out = new Dendrite();
    cudaError_t rval = cudaMemcpy(d_out, gpu_d, sizeof(Dendrite), cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    float Y = d_out->Y;
    delete d_out;
    return Y;
}