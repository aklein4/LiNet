#ifndef NEURON_H
#define NEURON_H

#include <cmath>
#include <stdlib.h>

const float P_OFFSET = 0.01;

typedef struct dendrite {

    /* Dynamic Values */
    float Y = static_cast<float>(0); // Value of the outgoing signal
    float p = static_cast<float>(-1); // single pole of the transfer function. Should always be < 0.
    
    float Y_offset = static_cast<float>(0); // Slightly offset value of the signal for calculating dp

    /* Control Theory-Based Values */
    float k = static_cast<float>(.125); // gain of the transfer function
    float delta_t = static_cast<float>(0.01); // time step between convolution updates
    float dp_offset = p * P_OFFSET;
    float dp_coefficient = 1.0/dp_offset;

    /* Static Computational Values */
    float gamma = static_cast<float>(std::exp(p * delta_t)); // "discounting coefficient" of transfer function. = e^p. Should always be in [0, 1)
    float gamma_offset = static_cast<float>(std::exp((p+dp_offset) * delta_t)); // discounting coefficient of the offset pole

    /* Keep track of the gradients to be applied. */
    float learning_rate = 0.00001;
    float dk = 0;
    float dp = 0;
    float grad_x = 0; 

    inline void apply_grads() {
        k += dk;
        dk = 0.0;

        p += __min(-p/2, dp);
        dp = 0.0;
        dp_offset = p * P_OFFSET;

        float gamma = static_cast<float>(std::exp(p * delta_t));
        float gamma_offset = static_cast<float>(std::exp((p+dp_offset) * delta_t));
    };

} Dendrite;

/* execute a convolutional transfer function */
inline __device__ __host__ float operator * (Dendrite& H, float X) {
    //if (X <= 0.0) X = 0.0;

    H.Y *= H.gamma;
    H.Y += X * H.k * H.delta_t;

    //H.Y_offset *= H.gamma_offset;
    //H.Y_offset += X * H.k * H.delta_t;

    //H.grad_x *= H.gamma;
    //if(X > 0.0) H.grad_x += H.delta_t; // ONLY WORKS FOR RELU ACTIVATION

    return H.Y;
};

/* Update the gradients of H using dX */
inline __device__ __host__ float operator ^ (Dendrite& H, float dX) {
    H.dk += (H.Y / H.k) * dX;
    H.dp += (H.Y_offset - H.Y) * H.dp_coefficient * dX;

    return H.grad_x * dX;
};

#endif