#ifndef NEURON_MATH_H
#define NEURON_MATH_H

#include "neuron.h"

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

#endif