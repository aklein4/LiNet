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

#endif