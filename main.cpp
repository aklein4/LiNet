
#include "choicenet.h"
#include <vector>
#include <iostream>

int main() {

    ChoiceNet net = ChoiceNet();
    for (int i=0; i<10; i++) {
        std::cout << net.update(1.0) << std::endl;
    }
    
}