
#include "choicenet.h"
#include <vector>
#include <iostream>

#include "dim_cuda.h"

int main() {

    size_t test_size = test_size;

    float* x_setter = new float[test_size];
    for (int i=0; i<test_size; i++) {
        x_setter[i] = i;
    }
    gpu::Vector1D<float> x(test_size);
    x.write(x_setter);
    delete[] x_setter;

    gpu::Matrix2D<float> A(test_size, test_size);
    for (int i=0; i<test_size; i++) {
        for (int j=0; j<test_size; j++) {
            A[j][i] = i + j;
        }
    }

    gpu::Vector1D<float> *y = gpu::matMulti<float, float>(A, x);
    float* y_out = new float[test_size];
    y.read(y_out);
    
    for (size_t i=0; i<test_size; i++) {
        std::cout << y[i] << std::endl;
    }

    delete[] y_out;

}