
#include "chunk.h"

Chunk::Chunk(size_t num_layers, size_t layer_size, int input_size, int output_size) {
    // dimensions
    num_layers_ = num_layers;
    layer_size_ = layer_size;
    input_size_ = (input_size < 0 ? layer_size_ : input_size);
    output_size_ = (output_size < 0 ? layer_size_ : output_size);

    // io buffers
    input_buffer_ = new float[input_size_];
    output_buffer_ = new float[output_size_];

    // activation layers
    activations_ = Matrix2D<float>(num_layers_, layer_size_);

    // transfer matrices
    transfers_ = Matrix3D<Dendrite>(num_layers_, layer_size_, layer_size_);
    input_transfer_ = Matrix2D<Dendrite>(layer_size_, input_size_);
    output_transfer_ = Matrix2D<Dendrite>(output_size_, layer_size_);
}

Chunk::~Chunk() {
    delete[] input_buffer_;
    delete[] output_buffer_;
}