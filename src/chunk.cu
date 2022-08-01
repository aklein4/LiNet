
#include "chunk.h"
#include "gpu_types.cuh"
#include "gpu_math.cuh"

Chunk::Chunk(size_t num_layers, size_t layer_size, int input_size, int output_size):
    // io buffers
    input_buffer_(input_size < 0 ? layer_size : input_size),
    output_buffer_(output_size < 0 ? layer_size : output_size),
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
    output_size_ = (output_size < 0 ? layer_size_ : output_size);

    // set internal activation layers to all zero
    clear_matrix_(activations_);

    // initialize transfer functions with all default contructions
    int coef = layer_size_;
    if (input_size_ > coef) coef = input_size_;
    if (output_size_ > coef) coef = output_size_;
    int writer_size = num_layers_ * coef * coef;
    Synapse* synapse_writer = new Synapse[writer_size];
    for (int i=0; i<writer_size; i++) synapse_writer[i] = Synapse();

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


void Chunk::forward_pass() {
    // input to first activation
    gpu::matMulti<Synapse, float, float>(input_transfer_, input_buffer_, activations_.vec(0), static_cast<float>(0.0));

    // internal activations
    for (int i=0; i < num_layers_-1; i++) {
        gpu::matMulti<Synapse, float, float>(transfers_[i], activations_.vec(i), activations_.vec(i+1), static_cast<float>(0.0));
    }

    // last activation to output
    gpu::matMulti<Synapse, float, float>(output_transfer_, activations_.vec(num_layers_-1), output_buffer_, static_cast<float>(0.0));
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