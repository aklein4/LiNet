#ifndef MEMQ_H
#define MEMQ_H

#include "neuron.cuh"
#include "gpu_types.cuh"
#include "vector"

class MemQNet {
    public:
        /* Create a new chunk with the specified dimensions.
         * Activation and transfer functions will be stored in 
         * matrices that handle their own memory. 
         * \param[in] num_layers The number of HIDDEN layers in the chunk
         * \param[in] layer_size The size of each layer vector's size
         * \param[in] input_size The length of the input vector, default=-1 sets to layer_size
         * \param[in] outnput_size The length of the outnput vector, default=-1 sets to layer_size
         * */
        MemQNet(float discount, size_t num_layers, size_t layer_size, int input_size=-1, int output_size=-1);
        /* Input and output buffers are deleted. Layer and Transfer matrices handle their own memory. */
        ~MemQNet();

        // execute a forward pass through the network
        int step(float prev_reward=0);

        /* Write the contents of the buffer into the input vector.
         * \param[in] buf buffer to copy from. */
        void set_input(float* buf);

    private:
        // dimensions
        size_t num_layers_;
        size_t layer_size_;
        size_t input_size_;
        size_t output_size_;

        // data buffers that can be read/written for communication
        gpu::Vector1D<float> input_buffer_;
        gpu::Vector1D<float> output_buffer_;

        // Contain the activation values of the internal layers
        gpu::Matrix2D<float> activations_;

        // Contains the transfer functions between layers, square
        // transfers_[i] is the transfer matrix whose INPUT vector is activations_[i]
        gpu::Matrix3D<float> weights_;
        // input and output may be different size from internal layers
        // may not be square
        gpu::Matrix2D<float> input_weights_;
        gpu::Matrix2D<float> output_weights_;

        // discount value for Q-learning [0, 1]
        float discount_;
        // epsilon value for greedy epsilon exploration
        float epsilon_;
        // whether the previous step was a maximum for modified greedy epsilon
        bool prev_max_;

        /* fill a vector with all zeroes */ 
        void clear_vector_(gpu::Vector1D<float> &vec);
        /* fill a 2D matrix with all zeroes */
        void clear_matrix_(gpu::Matrix2D<float> &mat);

        void initialize_weights_();
};

#endif