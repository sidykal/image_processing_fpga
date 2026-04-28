#include <iostream>
#include <cstdlib>
#include <ctime>
#include "test_lenet.h"

int main() {
    // Input buffer (matches HLS interface)
    bus_word_t input_words[INPUT_WORDS];

    // Seed randomness (optional)
    std::srand(0);

    // --------------------------------------------------
    // Fill input with dummy data (-128 to 127)
    // --------------------------------------------------
    for (int i = 0; i < INPUT_WORDS; i++) {
        bus_word_t word = 0;

        for (int b = 0; b < 4; b++) {
            int8_t val = (std::rand() % 256) - 128;
            word |= ((uint32_t)((uint8_t)val) << (8 * b));
        }

        input_words[i] = word;
    }

    // --------------------------------------------------
    // Run inference
    // --------------------------------------------------
    int predicted_class = -1;

    lenet_predict(input_words, &predicted_class);

    // --------------------------------------------------
    // Output result
    // --------------------------------------------------
    std::cout << "Predicted class: " << predicted_class << std::endl;

    return 0;
}