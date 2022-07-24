#ifndef CHUNK_H
#define CHUNK_H

#include "neuron.h"
#include "matrix.h"

class Chunk {
    public:
        /* Create a new chunk with the specified dimensions.
         * Activation and transfer functions will be stored in 
         * matrices that handle their own memory. 
         * \param[in] num_layers The number of HIDDEN layers in the chunk
         * \param[in] layer_size The size of each layer vector's size
         * \param[in] input_size The length of the input vector, default=-1 sets to layer_size
         * \param[in] outnput_size The length of the outnput vector, default=-1 sets to layer_size
         * */
        Chunk(size_t num_layers, size_t layer_size, int input_size=-1, int output_size=-1);
        /* Input and output buffers are deleted. Layer and Transfer matrices handle their own memory. */
        ~Chunk();

    private:
        // dimensions
        size_t num_layers_;
        size_t layer_size_;
        size_t input_size_;
        size_t output_size_;

        // data buffers that can be read/written for communication
        float* input_buffer_;
        float* output_buffer_;

        // Contain the activation values of the internal layers
        Matrix2D<float> activations_;

        // Contains the transfer functions between layers, square
        // transfers_[i] is the transfer matrix whose INPUT vector is activations_[i]
        Matrix3D<Dendrite> transfers_;
        // input and output may be different size from internal layers
        // may not be square
        Matrix2D<Dendrite> input_transfer_;
        Matrix2D<Dendrite> output_transfer_;
};

#endif