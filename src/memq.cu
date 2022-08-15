
#include "memq.h"
#include "gpu_types.cuh"
#include "gpu_math.cuh"
#include <random>
#include <math.h>

MemQNet::MemQNet(float discount, size_t num_layers, size_t layer_size, int input_size, int output_size):
    // io buffers
    input_buffer_((input_size < 0 ? layer_size : input_size)+1),
    output_buffer_((output_size < 0 ? layer_size : output_size)),
    // internal activations
    activations_(num_layers, layer_size),
    // transfer matrices
    weights_(num_layers - 1, layer_size, layer_size+1),
    input_weights_(layer_size, (input_size < 0 ? layer_size : input_size)+1),
    output_weights_((output_size < 0 ? layer_size : output_size), layer_size+1)
    {

    // dimensions
    num_layers_ = num_layers;
    layer_size_ = layer_size;
    input_size_ = (input_size < 0 ? layer_size_ : input_size);
    output_size_ = (output_size < 0 ? layer_size_ : output_size);

    // set internal activation layers to all zero
    clear_matrix_(activations_);

    // initialize random weights
    initialize_weights_();

    // the end of every buffer should be 1 for constant offset
    input_buffer_[input_size_].set(1.0);
    output_buffer_[output_size_].set(1.0);
    for (int k=0; k < num_layers_; k++) {
        activations_[k][layer_size_].set(1.0);
    }
}

MemQNet::~MemQNet() {
}


void MemQNet::initialize_weights_() {
    // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    std::default_random_engine generator;

    // initialize input weights
    std::normal_distribution <float> input_rand(0.0, std::sqrt(2.0/input_weights_.width()));
    for (int i=0; i < input_weights_.width(); i++) {
        for (int j=0; j < input_weights_.height(); j++) {
            input_weights_.vec(j)[i].set(input_rand(generator));
        }
    }

    // initialize output weights
    std::normal_distribution <float> output_rand(0.0, std::sqrt(2.0/output_weights_.width()));
    for (int i=0; i < output_weights_.width(); i++) {
        for (int j=0; j < output_weights_.height(); j++) {
            output_weights_.vec(j)[i].set(output_rand(generator));
        }
    }

    // initialize layer weights
    std::normal_distribution <float> layer_rand(0.0, std::sqrt(2.0/weights_.width()));
    for (int k=0; k < weights_.num(); k++) {
        for (int i=0; i < weights_.width(); i++) {
            for (int j=0; j < weights_.height(); j++) {
                weights_[k].vec(j)[i].set(layer_rand(generator));
            }
        }
    }
}


void MemQNet::set_input(float* buf) {
    assert(buf != NULL);

    input_buffer_.write(buf);
    input_buffer_[input_size_].set(1.0);
};


void MemQNet::step(float prev_reward) {
    // input to first activation
    gpu::matMulti_opt<Dendrite, float, float>(input_transfer_, input_buffer_, activations_.vec(0));

    // internal activations
    for (int i=0; i < num_layers_-1; i++) {
        gpu::matMulti_opt<Dendrite, float, float>(transfers_[i], activations_.vec(i), activations_.vec(i+1));
    }

    // last activation to output
    gpu::matMulti_opt<Dendrite, float, float>(output_transfer_, activations_.vec(num_layers_-1), output_buffer_);
}


/* Fill the buffer with zeroes. */
__global__ void gpu_clear_(float* buf, int size) {
    int i = blockIdx.x + threadIdx.x;
    if (i >= size) return;

    buf[i] = 0.0;
};
void MemQNet::clear_vector_(gpu::Vector1D<float> &vec) {
    // get block array
    int BLOCK_SIZE = 64;
    int array_size = vec.size() / BLOCK_SIZE;
    if (vec.size() % BLOCK_SIZE != 0) array_size ++;

    // call the device-bound clearing function
    gpu_clear_<<<array_size, BLOCK_SIZE>>>(vec.get_data(), vec.size());
}
void MemQNet::clear_matrix_(gpu::Matrix2D<float> &mat) {
    int size = mat.height() * mat.width();

    // get block array
    int BLOCK_SIZE = 16*16;
    int array_size = size / BLOCK_SIZE;
    if (size % BLOCK_SIZE != 0) array_size ++;

    // call the device-bound clearing function
    gpu_clear_<<<array_size, BLOCK_SIZE>>>(mat.get_data(), size);
}