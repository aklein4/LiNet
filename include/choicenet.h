#ifndef CHOICENET_H
#define CHOICENET_H

#include "neuron.cuh"
#include "chunk.h"

class ChoiceNet {
    public:
        ChoiceNet();
        ~ChoiceNet();

        float update(float x_in);

    private:
        Synapse* gpu_d; 
};

#endif