
#include "chunk.h"
#include "gpu_types.cuh"
#include "gpu_math.cuh"

#include <iostream>

Chunk::Chunk(size_t num_layers, size_t layer_size, int input_size, int output_size):
    // io buffers
    input_buffer_(input_size < 0 ? layer_size : input_size),
    output_buffer_(output_size < 0 ? layer_size : output_size),
    reward_buffer_(output_size < 0 ? layer_size : output_size),
    losses_(output_size < 0 ? layer_size : output_size),
    // internal activations
    activations_(num_layers, layer_size),
    // transfer matrices
    transfers_(num_layers - 1, layer_size, layer_size),
    input_transfer_(layer_size, (input_size < 0 ? layer_size : input_size)),
    output_transfer_((output_size < 0 ? layer_size : output_size), layer_size)
    {

    // dimensions
    num_layers_ = num_layers;
    layer_size_ = layer_size;
    input_size_ = (input_size < 0 ? layer_size_ : input_size);
    if (input_size_ % 32 != 0) std::cout << "ERROR: Chunk input size is not multiple of 32!" << std::endl;
    output_size_ = (output_size < 0 ? layer_size_ : output_size);

    // set internal activation layers to all zero
    clear_matrix_(activations_);

    // initialize transfer functions with all default contructions
    int coef = layer_size_;
    if (input_size_ > coef) coef = input_size_;
    if (output_size_ > coef) coef = output_size_;
    int writer_size = num_layers_ * coef * coef;
    Dendrite* synapse_writer = new Dendrite[writer_size];
    for (int i=0; i<writer_size; i++) synapse_writer[i] = Dendrite();

    input_transfer_.write(synapse_writer);
    output_transfer_.write(synapse_writer);
    transfers_.write(synapse_writer);

    delete[] synapse_writer;
}

Chunk::~Chunk() {
}


void Chunk::write(float* buf) {
    assert(buf != NULL);

    input_buffer_.write(buf);
};

void Chunk::read(float* buf) {
    assert(buf != NULL);

    output_buffer_.read(buf);
};


__global__ void calc_losses_(float* output, float* reward, float* loss) {
    loss[threadIdx.x] = reward[threadIdx.x] - output[threadIdx.x];
};
void Chunk::reward(float* buf) {
    assert(buf != NULL);

    reward_buffer_.read(buf);
    calc_losses_<<<1, output_size_>>>(output_buffer_.get_data(), reward_buffer_.get_data(), losses_.get_data());
};



void Chunk::forward() {

    // input to first activation
    gpu::matMulti<Dendrite, float, float>(input_transfer_, input_buffer_, activations_.vec(0), 0.0, false);

    // internal activations
    for (int i=0; i < num_layers_-1; i++) {
        gpu::matMulti<Dendrite, float, float>(transfers_[i], activations_.vec(i), activations_.vec(i+1), 0.0, false);
    }

    // last activation to output
    gpu::matMulti<Dendrite, float, float>(output_transfer_, activations_.vec(num_layers_-1), output_buffer_, 0.0, false);
}

void Chunk::backward() {
    
}

/* Fill the buffer with zeroes. */
__global__ void gpu_clear_(float* buf, int size) {
    int i = blockIdx.x + threadIdx.x;
    if (i >= size) return;

    buf[i] = 0.0;
};
void Chunk::clear_vector_(gpu::Vector1D<float> &vec) {
    // get block array
    int BLOCK_SIZE = 64;
    int array_size = vec.size() / BLOCK_SIZE;
    if (vec.size() % BLOCK_SIZE != 0) array_size ++;

    // call the device-bound clearing function
    gpu_clear_<<<array_size, BLOCK_SIZE>>>(vec.get_data(), vec.size());
}
void Chunk::clear_matrix_(gpu::Matrix2D<float> &mat) {
    int size = mat.height() * mat.width();

    // get block array
    int BLOCK_SIZE = 16*16;
    int array_size = size / BLOCK_SIZE;
    if (size % BLOCK_SIZE != 0) array_size ++;

    // call the device-bound clearing function
    gpu_clear_<<<array_size, BLOCK_SIZE>>>(mat.get_data(), size);
}