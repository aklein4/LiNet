
#include "cuda.h"

#include <stdlib.h>
#include <iostream>

int NUMBER = 320*1000;
int REPEAT = 10000;

int main() {

    float* a = new float[NUMBER];
    float* b = new float[NUMBER];
    float* c = new float[NUMBER];

    for (int i=0; i<NUMBER; i++) {
        a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        c[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    int profile = time_ms();
    for (int i=0; i < REPEAT; i++) {
        for (int n=0; n < NUMBER; n++) {
            c[n] = a[n] + b[n];
        }
    }
    profile = time_ms() - profile;
    std::cout << "CPU Time: " << profile << " ms" << std::endl;

    profile = add_vecs(a, b, c, NUMBER, REPEAT);
    std::cout << "CUDA Time: " << profile << " ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}