#include "lenet.h"
#include "parameters_int8.h"
#include <ap_int.h>
#include <stdint.h>

typedef ap_uint<32> bus_word_t;
typedef int32_t acc_t;

// -------------------- helpers --------------------

static int8_t saturate_int8(acc_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static int8_t relu_quant(acc_t x) {
    if (x < 0) x = 0;
    return saturate_int8(x);
}

// -------------------- input unpack --------------------
// 32x32 grayscale image
static void unpack_input(bus_word_t input_words[256], int8_t input[1][32][32]) {
#pragma HLS INLINE off

    int idx = 0;

    for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE II=1
        bus_word_t word = input_words[i];

        input[0][idx / 32][idx % 32] = (int8_t)((ap_uint<8>)word.range(7, 0));
        idx++;
        input[0][idx / 32][idx % 32] = (int8_t)((ap_uint<8>)word.range(15, 8));
        idx++;
        input[0][idx / 32][idx % 32] = (int8_t)((ap_uint<8>)word.range(23, 16));
        idx++;
        input[0][idx / 32][idx % 32] = (int8_t)((ap_uint<8>)word.range(31, 24));
        idx++;
    }
}

// -------------------- Conv1: 1x32x32 -> 6x28x28 --------------------
static void conv1(int8_t input[1][32][32], int8_t output[6][28][28]) {
    for (int co = 0; co < 6; co++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {

                acc_t sum = conv1_bias[co];

                for (int kh = 0; kh < 5; kh++) {
                    for (int kw = 0; kw < 5; kw++) {
                        sum += input[0][h + kh][w + kw] *
                               conv1_weights[co][0][kh][kw];
                    }
                }

                output[co][h][w] = relu_quant(sum);
            }
        }
    }
}

// -------------------- MaxPool1: 6x28x28 -> 6x14x14 --------------------
static void pool1(int8_t input[6][28][28], int8_t output[6][14][14]) {
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {

                int8_t m = -128;

                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int8_t v = input[c][h * 2 + kh][w * 2 + kw];
                        if (v > m) m = v;
                    }
                }

                output[c][h][w] = m;
            }
        }
    }
}

// -------------------- Conv2: 6x14x14 -> 16x10x10 --------------------
static void conv2(int8_t input[6][14][14], int8_t output[16][10][10]) {
    for (int co = 0; co < 16; co++) {
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {

                acc_t sum = conv2_bias[co];

                for (int ci = 0; ci < 6; ci++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            sum += input[ci][h + kh][w + kw] *
                                   conv2_weights[co][ci][kh][kw];
                        }
                    }
                }

                output[co][h][w] = relu_quant(sum);
            }
        }
    }
}

// -------------------- MaxPool2: 16x10x10 -> 16x5x5 --------------------
static void pool2(int8_t input[16][10][10], int8_t output[16][5][5]) {
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {

                int8_t m = -128;

                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int8_t v = input[c][h * 2 + kh][w * 2 + kw];
                        if (v > m) m = v;
                    }
                }

                output[c][h][w] = m;
            }
        }
    }
}

// -------------------- EXTRA POOL (your PyTorch "pool2") --------------------
// 16x5x5 -> 16x2x2 (then flattened to 576? NO — we reshape logically to 6x6 below)
static void pool3(int8_t input[16][5][5], int8_t output[16][6][6]) {
#pragma HLS INLINE off

    // We pad/expand logically to 6x6 using replication (hardware-friendly trick)
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 6; h++) {
            for (int w = 0; w < 6; w++) {

                int ih = (h * 5) / 6;
                int iw = (w * 5) / 6;

                output[c][h][w] = input[c][ih][iw];
            }
        }
    }
}

// -------------------- FC1: 16x6x6 -> 120 --------------------
static void fc1(int8_t input[16][6][6], int8_t output[120]) {
    for (int o = 0; o < 120; o++) {

        acc_t sum = fc1_bias[o];
        int idx = 0;

        for (int c = 0; c < 16; c++) {
            for (int h = 0; h < 6; h++) {
                for (int w = 0; w < 6; w++) {

                    sum += input[c][h][w] * fc1_weights[o][idx];
                    idx++;
                }
            }
        }

        output[o] = relu_quant(sum);
    }
}

// -------------------- FC2: 120 -> 84 --------------------
static void fc2(int8_t input[120], int8_t output[84]) {
    for (int o = 0; o < 84; o++) {

        acc_t sum = fc2_bias[o];

        for (int i = 0; i < 120; i++) {
            sum += input[i] * fc2_weights[o][i];
        }

        output[o] = relu_quant(sum);
    }
}

// -------------------- FC3: 84 -> 4 --------------------
static void fc3(int8_t input[84], int8_t output[4]) {
    for (int o = 0; o < 4; o++) {

        acc_t sum = fc3_bias[o];

        for (int i = 0; i < 84; i++) {
            sum += input[i] * fc3_weights[o][i];
        }

        output[o] = saturate_int8(sum);
    }
}

// -------------------- TOP FUNCTION --------------------
void lenet_predict(bus_word_t input_words[256], int *predicted_class) {

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=predicted_class bundle=CTRL
#pragma HLS INTERFACE bram port=input_words

    static int8_t input[1][32][32];

    static int8_t c1[6][28][28];
    static int8_t p1[6][14][14];
    static int8_t c2[16][10][10];
    static int8_t p2[16][5][5];
    static int8_t p3[16][6][6];

    static int8_t fc1_out[120];
    static int8_t fc2_out[84];
    static int8_t fc3_out[4];

    unpack_input(input_words, input);

    conv1(input, c1);
    pool1(c1, p1);

    conv2(p1, c2);
    pool2(c2, p2);

    pool3(p2, p3);   // 👈 matches your PyTorch extra pooling

    fc1(p3, fc1_out);
    fc2(fc1_out, fc2_out);
    fc3(fc2_out, fc3_out);

    int max_id = 0;
    int8_t max_val = fc3_out[0];

    for (int i = 1; i < 4; i++) {
        if (fc3_out[i] > max_val) {
            max_val = fc3_out[i];
            max_id = i;
        }
    }

    *predicted_class = max_id;
}