#include <omp.h>
#include <iostream>
#include <arm_neon.h>
#include <cmath>
#include <chrono>
#include <string.h>
#include "ffth_aarch64.h"

using namespace ffth_aarch64;

typedef std::chrono::_V2::system_clock::time_point TIME_POINT;

namespace ffth_aarch64 
{
    void debug_print(float *data, uint32_t size) {
        for (uint32_t i = 0; i < size; ++i){
            printf("%u: %.4f\n", (i + 1), data[i]);
        }
    }

    TIME_POINT now_() { 
        return std::chrono::high_resolution_clock::now();
    }

    void time_ms(TIME_POINT start_, TIME_POINT end_) {
        std::chrono::duration<double, std::milli> out1 = end_ - start_;
        std::chrono::duration<double, std::micro> out2 = end_ - start_;
        printf("Time: %f (ms), %f (us)\n", out1.count(), out2.count());
    }

    inline float fmadd(const float *in1, const float *in2, uint32_t size) {
        uint32_t cntBlock = size / NEONSIZE;
        uint32_t cntRem = size - cntBlock * NEONSIZE;
        float32x4_t resultData, loadData1, loadData2;
        resultData[0] = 0;
        resultData[1] = 0;
        resultData[2] = 0;
        resultData[3] = 0;
        for (uint32_t i = 0; i < cntBlock; ++i) {
            loadData1 = vld1q_f32(in1);
            loadData2 = vld1q_f32(in2);
            resultData = vfmaq_f32(resultData, loadData1, loadData2);
            in1 += NEONSIZE;
            in2 += NEONSIZE;
        }
        float out = resultData[0] + resultData[1] + resultData[2] + resultData[3];
        for (uint32_t i = 0; i < cntRem; ++i) {
            out += in1[i] * in2[i];
        }
        return out;
    }

    ffth::ffth(FFTH_PARAM_S *ffth_param)
    {
        win_inc = ffth_param->win_inc;
        win_shift = ffth_param->win_shift;
        win_len = ffth_param->win_len;
        fft_len = ffth_param->fft_len;
        if (ffth_param->mode == STAND) {
            ffth_param->in_size = (win_inc - 1) * win_shift + fft_len;
            ffth_param->out_size = win_inc * (fft_len + 2);
            ffth_param->kernel = new float[(fft_len + 2) * fft_len]();
            ffth_param->in = new float[ffth_param->in_size]();
            ffth_param->out = new float[ffth_param->out_size]();
            
            uint32_t imag_start = (fft_len / 2 + 1) * fft_len;
            for (uint32_t i = 0; i < (fft_len / 2 + 1); ++i) {
                for (uint32_t j = 0; j < fft_len; ++j) {
                    ffth_param->kernel[i * fft_len + j] = cos(2 * M_PI * j * i / fft_len);
                    ffth_param->kernel[i * fft_len + j + imag_start] = -sin(2 * M_PI * j * i / fft_len);
                }
            }

            if (ffth_param->hamming_en) {
                ffth_param->hamming = new float[fft_len]();
                for (uint32_t i = 0; i < win_len; ++i) {
                    ffth_param->hamming[i + (fft_len - win_len) / 2] = 0.54f - 0.46f * cos(2 * M_PI * i / (win_len - 1));
                }
                for (uint32_t i = 0; i < (fft_len + 2); ++i) {
                    for (uint32_t j = 0; j < fft_len; ++j) {
                        ffth_param->kernel[i * fft_len + j] *= ffth_param->hamming[j];
                    }
                }
            }
        } else {
            ffth_param->in_size = win_inc * (fft_len + 2);
            ffth_param->out_size = (win_inc - 1) * win_shift + fft_len;
            ffth_param->kernel = new float[(fft_len + 2) * fft_len]();
            ffth_param->in = new float[ffth_param->in_size]();
            ffth_param->out = new float[ffth_param->out_size]();

            uint32_t tmp_1 = fft_len + 2;
            uint32_t tmp_2 = tmp_1 / 2;
            for (uint32_t i = 0; i < fft_len; ++i) {
                for (uint32_t j = 0; j < tmp_2; ++j) {
                    ffth_param->kernel[i * tmp_1 + j] = cos(2 * M_PI * j * i / fft_len) / fft_len;
                    ffth_param->kernel[i * tmp_1 + j + tmp_2] = -sin(2 * M_PI * j * i / fft_len) / fft_len;
                }
                for (uint32_t j = 0; j < (tmp_2 - 2); ++j) {
                    ffth_param->kernel[i * tmp_1 + j + 1] += cos(2 * M_PI * (fft_len - 1 - j) * i / fft_len) / fft_len;
                    ffth_param->kernel[i * tmp_1 + j + tmp_2 + 1] += sin(2 * M_PI * (fft_len - 1 - j) * i / fft_len) / fft_len;
                }
            }
        }

        this->ffth_param = ffth_param;
    }
    ffth::~ffth()
    {
        delete[]ffth_param->hamming;
        delete[]ffth_param->kernel;
        delete[]ffth_param->in;
        delete[]ffth_param->out;
    }

    int ffth::compute() {
        float *in = ffth_param->in;
        float *out = ffth_param->out;
        float *kernel = ffth_param->kernel;

        if (ffth_param->mode == STAND) {
            uint32_t out_w = (fft_len + 2);
            uint32_t size = win_inc * out_w;
            #pragma omp parallel for num_threads(THREADSIZE)
            for (uint32_t idx = 0; idx < size; ++idx) {
                uint32_t i, j;
                i = idx / out_w;
                j = idx - i * out_w;
                out[idx] = fmadd(in + i * win_shift, kernel + j * fft_len, fft_len);
            }
            return 0;
        }
        
        memset(out, 0, ffth_param->out_size * sizeof(float));
        uint32_t in_w = fft_len + 2;
        for (uint32_t i = 0; i < win_inc; ++i) {
            #pragma omp parallel for num_threads(THREADSIZE)
            for (uint32_t j = 0; j < fft_len; ++j) {
                out[i * win_shift + j] += fmadd(in + i * in_w, kernel + j * in_w, in_w);
            }
        }
        return 0;
    }
} // ffth_aarch64