#ifndef FFTH_AARCH64_H
#define FFTH_AARCH64_H
#define NEONSIZE 4
#define THREADSIZE 4

namespace ffth_aarch64 
{   
    typedef enum h_FFTH_MODE_E 
    {
        STAND,
        INVERSE
    } FFTH_MODE_E;

    /**
     * @param hamming_en: manual
     * @param mode: manual
     * @param win_inc: manual
     * @param win_shift: manual
     * @param win_len: manual
     * @param fft_len: manual
     * @param other: auto
     */
    typedef struct h_FFTH_PARAM_S
    {
        bool hamming_en;
        FFTH_MODE_E mode;
        uint32_t win_inc;
        uint32_t win_shift;
        uint32_t win_len;
        uint32_t fft_len;
        uint32_t in_size;
        uint32_t out_size;
        float *hamming;
        float *kernel;
        float *in;
        float *out;
        public:
        h_FFTH_PARAM_S() 
        {
            hamming_en = false;
            mode = STAND;
            win_inc = 1;
            win_shift = 0;
        }
    } FFTH_PARAM_S;
    
    class ffth
    {   
        private:
        FFTH_PARAM_S *ffth_param;
        uint32_t win_inc;
        uint32_t win_shift;
        uint32_t win_len;
        uint32_t fft_len;

        public:
        ffth(FFTH_PARAM_S *ffth_param);
        ~ffth();

        int compute();
    };

} // namespace ffth_aarch64

#endif // FFTH_AARCH64_H