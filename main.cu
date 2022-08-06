
#include "chunk.h"
#include "timer.h"

#include <iostream>

int main() {
    size_t test_size = 1024;

    // create input buffer and write values to it
    float* in_buf = new float;
    *in_buf = 1;

    // create another buffer to read output into
    float* out_buf = new float;
    *out_buf = 0.0;

    // create a chunk
    Chunk chunk(10, test_size, 1, 1);
    chunk.write(in_buf);

    // update the chunk
    Timer timer(TIME_UNIT::ms);
    int i=0;
    while(timer.get() < 1000) {
        //chunk.read(out_buf);
        //std::cout << t << ": " << *out_buf << std::endl;
        chunk.forward_pass();
        i++;
    }

    // print values
    std::cout << "Iterations per second: " << i << std::endl;
    //timer.print("CUDA:");

    // clean up
    delete in_buf;
    delete out_buf;

    return 0;
};