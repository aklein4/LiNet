#ifndef NEURON_H
#define NEURON_H

typedef struct {

    /* Dynamic Values */
    float Y; // Value of the outgoing signal
    float approx_plus; // upper approximation of dx/dp
    float approx_minus; // lower approximation of dx/dp

    /* Control Theory-Based Values */
    float p; // single pole of the transfer function. Should always be < 0.
    float k; // gain of the transfer function
    float p_plus; // higher pole of the dx/dp approximation
    float p_minus; // lower pole of the dx/dp approximation
    float approx_diff; // difference between p approximation poles and p
    float approx_multi; // multiplier to scale dx/dp

    /* Static Computational Values */
    float gamma; // "discounting coefficient" of transfer function. = e^p. Should always be in [0, 1)
    float gamma_plus; // upper approximation of gamma
    float gamma_minus; // lower approximation of gamma

    /* Pointers */
    float* X; // pointer to the incoming signal

} dendrite;



#endif