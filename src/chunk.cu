
#include "chunk.cuh"

Chunk::Chunk(size_t num_layers, size_t layer_size, int input_size, int output_size):
    // activation layers
    activations_(num_layers, layer_size),
    // transfer matrices
    transfers_(num_layers, layer_size, layer_size),
    input_transfer_(layer_size, (input_size < 0 ? layer_size : input_size)),
    output_transfer_((output_size < 0 ? layer_size : output_size), layer_size)
    {
    
    // dimensions
    num_layers_ = num_layers;
    layer_size_ = layer_size;
    input_size_ = (input_size < 0 ? layer_size_ : input_size);
    output_size_ = (output_size < 0 ? layer_size_ : output_size);

    // io buffers
    input_buffer_ = new float[input_size_];
    output_buffer_ = new float[output_size_];
}

Chunk::~Chunk() {
    delete[] input_buffer_;
    delete[] output_buffer_;
}


int Chunk::forward_pass() {
    return 0;
}