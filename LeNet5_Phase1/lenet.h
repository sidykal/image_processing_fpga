#ifndef LENET_H
#define LENET_H

#include <stdint.h>
#include <ap_int.h>

// -------------------- Input --------------------
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 1

// -------------------- Conv Layer 1 --------------------
// 1 → 6 channels, 5x5 kernel
#define C1_H 28
#define C1_W 28
#define C1_CH 6
#define C1_K 5

// -------------------- MaxPool 1 --------------------
// 2x2 pooling
#define P1_H 14
#define P1_W 14
#define P1_CH 6
#define P1_K 2

// -------------------- Conv Layer 2 --------------------
// 6 → 16 channels, 5x5 kernel
#define C2_H 10
#define C2_W 10
#define C2_CH 16
#define C2_K 5

// -------------------- MaxPool 2 --------------------
// 2x2 pooling
#define P2_H 5
#define P2_W 5
#define P2_CH 16
#define P2_K 2

// -------------------- EXTRA POOL (matches PyTorch pool2) --------------------
// Forces final spatial size to 6x6 (as required by FC layer: 16*6*6)
#define P3_H 6
#define P3_W 6
#define P3_CH 16

// -------------------- Fully Connected Layers --------------------
#define FC1_INPUT  (16 * 6 * 6)   // = 576
#define FC1_UNITS  120
#define FC2_UNITS  84
#define OUTPUT_CLASSES 4

// -------------------- Packed Input --------------------
// 32x32 image = 1024 int8 values
// 4 pixels per 32-bit word → 256 words
typedef ap_uint<32> bus_word_t;
#define INPUT_WORDS 256

// -------------------- Top Function --------------------
void lenet_predict(bus_word_t input_words[INPUT_WORDS], int *predicted_class);

#endif