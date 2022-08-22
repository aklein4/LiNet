
#include "chunk.h"
#include "timer.h"

#include <iostream>

int main() {
    size_t test_size = 256;

    // create input buffer and write values to it
    float* in_buf = new float[32];
    for (int i=0; i<32; i++) in_buf[i] = 0.0;

    // create another buffer to read output into
    float* out_buf = new float[32];
    *out_buf = 0.0;

    // create a chunk
    Chunk chunk(4, test_size, 32, 32);
    chunk.write(in_buf);

    // update the chunk
    Timer timer(TIME_UNIT::ms);
    int i=0;
    while(timer.get() < 1000) {
        //chunk.read(out_buf);
        //std::cout << t << ": " << *out_buf << std::endl;
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        chunk.forward();
        i+=10;
    }

    // print values
    std::cout << "Iterations per second: " << i << std::endl;
    //timer.print("CUDA:");

    // clean up
    delete[] in_buf;
    delete[] out_buf;

    return 0;
};