#include <iostream>
#include <fstream>
#include <cstdint>
#include "test_lenet.h"

int main() {
    bus_word_t input_words[INPUT_WORDS];

    std::ifstream file("input.txt");

    if (!file.is_open()) {
        std::cerr << "Error: could not open input.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < INPUT_WORDS; i++) {
        uint32_t word;
        file >> word;
        input_words[i] = word;
    }

    file.close();

    int predicted_class = -1;

    lenet_predict(input_words, &predicted_class);

    std::cout << "Predicted class: " << predicted_class << std::endl;

    return 0;
}