
#include "chunk.h"
#include "timer.h"

#include <iostream>

int main() {
    size_t test_size = 256;

    // create input buffer and write values to it
    float* in_buf = new float;
    *in_buf = 1.0;

    // create another buffer to read output into
    float* out_buf = new float;
    *out_buf = 0.0;

    // create a chunk
    Chunk chunk(16, test_size, 1, 1);
    chunk.write(in_buf);

    // update the chunk
    Timer timer(TIME_UNIT::us);
    chunk.forward_pass();

    // print values
    timer.print("CUDA:");
    chunk.read(out_buf);
    std::cout << *out_buf << std::endl;

    // clean up
    delete in_buf;
    delete out_buf;

    return 0;
};