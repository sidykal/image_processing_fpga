#include "test_lenet.h"
#include "parameters_int8.h"
#include <stdint.h>

#include <stdio.h>

#define CONV1_SHIFT 11
#define CONV2_SHIFT 10
#define FC1_SHIFT 9
#define FC2_SHIFT 9
#define FC3_SHIFT 8

static void print_stats_1d(const char* name, int8_t* data, int size) {
    int min_v = data[0];
    int max_v = data[0];
    int zeros = 0;
    int sat_pos = 0;
    int sat_neg = 0;

    for (int i = 0; i < size; i++) {
        int v = data[i];

        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v == 0) zeros++;
        if (v == 127) sat_pos++;
        if (v == -128) sat_neg++;
    }

    printf("%s: min=%d max=%d zeros=%d sat127=%d sat-128=%d total=%d\n",
           name, min_v, max_v, zeros, sat_pos, sat_neg, size);
}

typedef int32_t acc_t;

//////////////////////////////////////////////////////////////
// Quant helpers
//////////////////////////////////////////////////////////////
static int8_t saturate_int8(acc_t x) {
    if (x > 127)  return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static int8_t relu_quant(acc_t x) {
    if (x < 0) x = 0;
    return saturate_int8(x);
}

//////////////////////////////////////////////////////////////
// INPUT UNPACK: 1024 words → 1x64x64
//////////////////////////////////////////////////////////////
static void unpack_input(bus_word_t input_words[INPUT_WORDS],
                         int8_t input[INPUT_C][INPUT_H][INPUT_W]) {

#pragma HLS INLINE off

    int idx = 0;

    for (int i = 0; i < INPUT_WORDS; i++) {
#pragma HLS PIPELINE II=1
        bus_word_t word = input_words[i];

        for (int b = 0; b < 4; b++) {
            int8_t val = (int8_t)((word >> (8 * b)) & 0xFF);
            input[0][idx / INPUT_W][idx % INPUT_W] = val;
            idx++;
        }
    }
}

//////////////////////////////////////////////////////////////
// CONV1: 1x64x64 → 6x60x60
//////////////////////////////////////////////////////////////
static void conv1(int8_t in[INPUT_C][INPUT_H][INPUT_W],
                  int8_t out[C1_CH][C1_H][C1_W]) {

    for (int co = 0; co < C1_CH; co++) {
        for (int h = 0; h < C1_H; h++) {
            for (int w = 0; w < C1_W; w++) {

                acc_t sum = conv1_bias[co];

                for (int kh = 0; kh < C1_K; kh++) {
                    for (int kw = 0; kw < C1_K; kw++) {
                        sum += in[0][h+kh][w+kw] *
                               conv1_weights[co][0][kh][kw];
                    }
                }

                out[co][h][w] = relu_quant(sum >> CONV1_SHIFT);
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// POOL1: 60x60 → 30x30
//////////////////////////////////////////////////////////////
static void pool1(int8_t in[C1_CH][C1_H][C1_W],
                  int8_t out[S1_CH][S1_H][S1_W]) {

    for (int c = 0; c < S1_CH; c++) {
        for (int h = 0; h < S1_H; h++) {
            for (int w = 0; w < S1_W; w++) {

                int8_t m = -128;

                for (int kh = 0; kh < S1_K; kh++) {
                    for (int kw = 0; kw < S1_K; kw++) {
                        int8_t v = in[c][h*2+kh][w*2+kw];
                        if (v > m) m = v;
                    }
                }

                out[c][h][w] = m;
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// CONV2: 6x30x30 → 16x26x26
//////////////////////////////////////////////////////////////
static void conv2(int8_t in[S1_CH][S1_H][S1_W],
                  int8_t out[C2_CH][C2_H][C2_W]) {

    for (int co = 0; co < C2_CH; co++) {
        for (int h = 0; h < C2_H; h++) {
            for (int w = 0; w < C2_W; w++) {

                acc_t sum = conv2_bias[co];

                for (int ci = 0; ci < S1_CH; ci++) {
                    for (int kh = 0; kh < C2_K; kh++) {
                        for (int kw = 0; kw < C2_K; kw++) {
                            sum += in[ci][h+kh][w+kw] *
                                   conv2_weights[co][ci][kh][kw];
                        }
                    }
                }

                out[co][h][w] = relu_quant(sum >> CONV2_SHIFT);
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// POOL2: 26x26 → 13x13
//////////////////////////////////////////////////////////////
static void pool2(int8_t in[C2_CH][C2_H][C2_W],
                  int8_t out[S2_CH][S2_H][S2_W]) {

    for (int c = 0; c < S2_CH; c++) {
        for (int h = 0; h < S2_H; h++) {
            for (int w = 0; w < S2_W; w++) {

                int8_t m = -128;

                for (int kh = 0; kh < S2_K; kh++) {
                    for (int kw = 0; kw < S2_K; kw++) {
                        int8_t v = in[c][h*2+kh][w*2+kw];
                        if (v > m) m = v;
                    }
                }

                out[c][h][w] = m;
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// EXTRA POOL: 13x13 → 6x6  (CRITICAL)
//////////////////////////////////////////////////////////////
static void pool3(int8_t in[S2_CH][S2_H][S2_W],
                  int8_t out[S3_CH][S3_H][S3_W]) {

    for (int c = 0; c < S3_CH; c++) {
        for (int h = 0; h < S3_H; h++) {
            for (int w = 0; w < S3_W; w++) {

                int8_t m = -128;

                for (int kh = 0; kh < S3_K; kh++) {
                    for (int kw = 0; kw < S3_K; kw++) {
                        int8_t v = in[c][h*2+kh][w*2+kw];
                        if (v > m) m = v;
                    }
                }

                out[c][h][w] = m;
            }
        }
    }
}

//////////////////////////////////////////////////////////////
// FC1: 576 → 120
//////////////////////////////////////////////////////////////
static void fc1(int8_t in[S3_CH][S3_H][S3_W],
                int8_t out[FC1_UNITS]) {

    for (int o = 0; o < FC1_UNITS; o++) {
        acc_t sum = fc1_bias[o];

        int idx = 0;

        for (int c = 0; c < S3_CH; c++) {
            for (int h = 0; h < S3_H; h++) {
                for (int w = 0; w < S3_W; w++) {
                    sum += in[c][h][w] *
                           fc1_weights[o][idx++];
                }
            }
        }

        out[o] = relu_quant(sum >> FC1_SHIFT);
    }
}

//////////////////////////////////////////////////////////////
// FC2: 120 → 84
//////////////////////////////////////////////////////////////
static void fc2(int8_t in[FC1_UNITS],
                int8_t out[FC2_UNITS]) {

    for (int o = 0; o < FC2_UNITS; o++) {
        acc_t sum = fc2_bias[o];

        for (int i = 0; i < FC1_UNITS; i++) {
            sum += in[i] * fc2_weights[o][i];
        }

        out[o] = relu_quant(sum >> FC2_SHIFT);
    }
}

//////////////////////////////////////////////////////////////
// FC3: 84 → 4 (logits)
//////////////////////////////////////////////////////////////
static void fc3(int8_t in[FC2_UNITS],
                int8_t out[OUTPUT_CLASSES]) {

    for (int o = 0; o < OUTPUT_CLASSES; o++) {
        acc_t sum = fc3_bias[o];

        for (int i = 0; i < FC2_UNITS; i++) {
            sum += in[i] * fc3_weights[o][i];
        }

        out[o] = saturate_int8(sum >> FC3_SHIFT);
    }
}

//////////////////////////////////////////////////////////////
// TOP FUNCTION
//////////////////////////////////////////////////////////////
void lenet_predict(bus_word_t input_words[INPUT_WORDS],
                   int *predicted_class) {

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=predicted_class bundle=CTRL
#pragma HLS INTERFACE bram port=input_words

    static int8_t input[INPUT_C][INPUT_H][INPUT_W];
    static int8_t c1[C1_CH][C1_H][C1_W];
    static int8_t s1[S1_CH][S1_H][S1_W];
    static int8_t c2[C2_CH][C2_H][C2_W];
    static int8_t s2[S2_CH][S2_H][S2_W];
    static int8_t s3[S3_CH][S3_H][S3_W];
    static int8_t f1[FC1_UNITS];
    static int8_t f2[FC2_UNITS];
    static int8_t f3[OUTPUT_CLASSES];

    unpack_input(input_words, input);
    print_stats_1d("input", &input[0][0][0], INPUT_C * INPUT_H * INPUT_W);

    conv1(input, c1);
    print_stats_1d("c1", &c1[0][0][0], C1_CH * C1_H * C1_W);

    pool1(c1, s1);
    print_stats_1d("s1", &s1[0][0][0], S1_CH * S1_H * S1_W);

    conv2(s1, c2);
    print_stats_1d("c2", &c2[0][0][0], C2_CH * C2_H * C2_W);

    pool2(c2, s2);
    print_stats_1d("s2", &s2[0][0][0], S2_CH * S2_H * S2_W);

    pool3(s2, s3);
    print_stats_1d("s3", &s3[0][0][0], S3_CH * S3_H * S3_W);

    fc1(s3, f1);
    print_stats_1d("f1", f1, FC1_UNITS);

    fc2(f1, f2);
    print_stats_1d("f2", f2, FC2_UNITS);

    fc3(f2, f3);
    print_stats_1d("f3", f3, OUTPUT_CLASSES);

    printf("f3 logits:\n");
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        printf("class %d: %d\n", i, f3[i]);
    }

    // Argmax
    int max_id = 0;
    int8_t max_val = f3[0];

    for (int i = 1; i < OUTPUT_CLASSES; i++) {
        if (f3[i] > max_val) {
            max_val = f3[i];
            max_id = i;
        }
    }

    *predicted_class = max_id;
}