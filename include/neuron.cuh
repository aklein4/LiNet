#ifndef NEURON_H
#define NEURON_H

#include <cmath>

typedef struct dendrite {

    /* Dynamic Values */
    float Y = static_cast<float>(0); // Value of the outgoing signal
    //float approx_plus = static_cast<float>(0); // upper approximation of dx/dp
    //float approx_minus = static_cast<float>(0); // lower approximation of dx/dp

    /* Control Theory-Based Values */
    //float p = static_cast<float>(-1); // single pole of the transfer function. Should always be < 0.
    float k = static_cast<float>(.125); // gain of the transfer function
    //float p_plus = static_cast<float>(-0.99); // higher pole of the dx/dp approximation
    //float p_minus = static_cast<float>(-1.01); // lower pole of the dx/dp approximation
    //float approx_diff = static_cast<float>(0.01); // difference between p approximation poles and p
    //float approx_multi = static_cast<float>(50); // multiplier to scale dx/dp'
    float delta_t = static_cast<float>(0.01); // time step between convolution updates

    /* Static Computational Values */
    float gamma = static_cast<float>(std::exp(-1 * delta_t)); // "discounting coefficient" of transfer function. = e^p. Should always be in [0, 1)
    //float gamma_plus = static_cast<float>(std::exp(-0.99)); // upper approximation of gamma
    //float gamma_minus = static_cast<float>(std::exp(-1.01)); // lower approximation of gamma

} Dendrite;

/* execute a convolutional transfer function */
inline __device__ __host__ float operator * (float X, Dendrite& H) {
    H.Y *= H.gamma;
    H.Y += X * H.k * H.delta_t;
    return H.Y;
};

#endif