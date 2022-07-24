#ifndef CHOICENET_H
#define CHOICENET_H

#include "neuron.h"

class ChoiceNet {
    public:
        ChoiceNet();
        ~ChoiceNet();

        float update(float x_in);

    private:
        Dendrite* gpu_d; 
};

#endif