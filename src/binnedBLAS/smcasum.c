#include <stdlib.h>
#include <math.h>

#include "../config.h"
#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
import depositASum
import vectorizations
from src.common import blockSize
from scripts import terminal

code_block = generate.CodeBlock()
vectorizations.conditionally_include_vectorizations(code_block)
cog.out(str(code_block))

cog.outl()

cog.out(generate.generate(blockSize.BlockSize("smcasum", "N_block_MAX", 32, terminal.get_siendurance()//2, terminal.get_siendurance()//2, ["bench_rscasum_fold_{}".format(terminal.get_sidefaultfold())]), cog.inFile, args, params, mode))
]]]*/
#if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
  #include <immintrin.h>

#elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
  #include <emmintrin.h>

#else


#endif

#define N_block_MAX 256
//[[[end]]]

/**
 * @internal
 * @brief Add to manually specified binned single precision Y the absolute sum of complex single precision vector X
 *
 * Add to Y the binned sum of magnitudes of elements of X.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X complex single precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param priY Y's primary vector
 * @param incpriY stride within Y's primary vector (use every incpriY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_smcasum(const int fold, const int N, const void *X, const int incX, float *priY, const int incpriY, float *carY, const int inccarY){
  float amax_tmp[2];
  float amax;
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;
  float_complex_binned *asum = binned_cballoc(fold);
  binned_cbsetzero(fold, asum);

  const float *x = (const float*)X;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    binnedBLAS_camax_sub(N_block, x, incX, amax_tmp);
    amax = MAX(amax_tmp[0], amax_tmp[1]);

    if (isinf(amax) || isinf(priY[0])){
      for (j = 0; j < N_block; j++){
        priY[0] += fabsf(x[j * 2 * incX]);
        priY[0] += fabsf(x[j * 2 * incX + 1]);
      }
    }
    if (isnan(priY[0]) || isnan(asum[0]) || isnan(asum[1])){
      priY[0] += asum[0] + asum[1];
      free(asum);
      return;
    } else if (isinf(priY[0])){
      x += N_block * 2 * incX;
      continue;
    }

    if (deposits + N_block > binned_SBENDURANCE) {
      binned_cbrenorm(fold, asum);
      deposits = 0;
    }

    binned_cbsupdate(fold, amax, asum);

    /*[[[cog
      cog.out(generate.generate(depositASum.DepositASum(dataTypes.FloatComplex, "fold", "N_block", "x", "incX", "asum", 1), cog.inFile, args, params, mode))
      ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
        __m256 abs_mask_tmp;
        {
          __m256 tmp;
          tmp = _mm256_set1_ps(1);
          abs_mask_tmp = _mm256_set1_ps(-1);
          abs_mask_tmp = _mm256_xor_ps(abs_mask_tmp, tmp);
          tmp = _mm256_cmp_ps(tmp, tmp, 0);
          abs_mask_tmp = _mm256_xor_ps(abs_mask_tmp, tmp);
        }
        __m256 blp_mask_tmp;
        {
          __m256 tmp;
          blp_mask_tmp = _mm256_set1_ps(1.0);
          tmp = _mm256_set1_ps(1.0 + (FLT_EPSILON * 1.0001));
          blp_mask_tmp = _mm256_xor_ps(blp_mask_tmp, tmp);
        }
        __m256 cons_tmp; (void)cons_tmp;
        float cons_buffer_tmp[8] __attribute__((aligned(32))); (void)cons_buffer_tmp;
        unsigned int SIMD_daz_ftz_old_tmp = 0;
        unsigned int SIMD_daz_ftz_new_tmp = 0;


        switch(fold){
          case 2:
            {
              int i;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;

              s_0_0 = s_0_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum)));
              s_1_0 = s_1_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 2));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)asum))));
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)asum)) + 2));
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;
              __m256 s_2_0, s_2_1;

              s_0_0 = s_0_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum)));
              s_1_0 = s_1_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 2));
              s_2_0 = s_2_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 4));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    x_5 = _mm256_add_ps(x_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    x_3 = _mm256_add_ps(x_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    x_1 = _mm256_add_ps(x_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)asum))));
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)asum)) + 2));
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(((float*)asum)[5], ((float*)asum)[4], ((float*)asum)[5], ((float*)asum)[4], ((float*)asum)[5], ((float*)asum)[4], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)asum)) + 4));
              s_2_0 = _mm256_add_ps(s_2_0, _mm256_sub_ps(s_2_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0;
              __m256 s_0_0;
              __m256 s_1_0;
              __m256 s_2_0;
              __m256 s_3_0;

              s_0_0 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum)));
              s_1_0 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 2));
              s_2_0 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 4));
              s_3_0 = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + 6));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_4 = _mm256_add_ps(x_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_5 = _mm256_add_ps(x_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_5, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_2 = _mm256_add_ps(x_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_3 = _mm256_add_ps(x_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_1 = _mm256_add_ps(x_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    x_0 = _mm256_add_ps(x_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], ((float*)asum)[1], ((float*)asum)[0], 0, 0));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], ((float*)asum)[3], ((float*)asum)[2], 0, 0));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(((float*)asum)[5], ((float*)asum)[4], ((float*)asum)[5], ((float*)asum)[4], ((float*)asum)[5], ((float*)asum)[4], 0, 0));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_3_0 = _mm256_sub_ps(s_3_0, _mm256_set_ps(((float*)asum)[7], ((float*)asum)[6], ((float*)asum)[7], ((float*)asum)[6], ((float*)asum)[7], ((float*)asum)[6], 0, 0));
              _mm256_store_ps(cons_buffer_tmp, s_3_0);
              ((float*)asum)[6] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)asum)[7] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0;
              __m256 s_0;
              __m256 s_buffer[binned_SBMAXFOLD];

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = (__m256)_mm256_broadcast_sd((double *)(((float*)asum) + (j * 2)));
              }

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 32 <= N_block; i += 32, x += 64){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);
                    x_6 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 48), abs_mask_tmp);
                    x_7 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 56), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_4 = _mm256_add_ps(x_4, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_5, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_5 = _mm256_add_ps(x_5, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_6 = _mm256_add_ps(x_6, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_6, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_7, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_7 = _mm256_add_ps(x_7, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 32 <= N_block; i += 32, x += 64){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 40), abs_mask_tmp);
                    x_6 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 48), abs_mask_tmp);
                    x_7 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 56), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_4 = _mm256_add_ps(x_4, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_4, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_5, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_5 = _mm256_add_ps(x_5, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_6 = _mm256_add_ps(x_6, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_6, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_7, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_7 = _mm256_add_ps(x_7, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 24), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_loadu_ps(((float*)x) + 8), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_loadu_ps(((float*)x)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm256_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 32 <= N_block; i += 32, x += (incX * 64)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);
                    x_6 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 54) + 1)], ((float*)x)[(incX * 54)], ((float*)x)[((incX * 52) + 1)], ((float*)x)[(incX * 52)], ((float*)x)[((incX * 50) + 1)], ((float*)x)[(incX * 50)], ((float*)x)[((incX * 48) + 1)], ((float*)x)[(incX * 48)]), abs_mask_tmp);
                    x_7 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 62) + 1)], ((float*)x)[(incX * 62)], ((float*)x)[((incX * 60) + 1)], ((float*)x)[(incX * 60)], ((float*)x)[((incX * 58) + 1)], ((float*)x)[(incX * 58)], ((float*)x)[((incX * 56) + 1)], ((float*)x)[(incX * 56)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_4 = _mm256_add_ps(_mm256_add_ps(x_4, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_4 = _mm256_add_ps(x_4, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_4, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_5 = _mm256_add_ps(_mm256_add_ps(x_5, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_5, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_5 = _mm256_add_ps(x_5, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_6 = _mm256_add_ps(x_6, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_6, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_7, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_7 = _mm256_add_ps(x_7, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_ps(s_0, q_0);
                    x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 32 <= N_block; i += 32, x += (incX * 64)){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_4 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)], ((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_5 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)], ((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);
                    x_6 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 54) + 1)], ((float*)x)[(incX * 54)], ((float*)x)[((incX * 52) + 1)], ((float*)x)[(incX * 52)], ((float*)x)[((incX * 50) + 1)], ((float*)x)[(incX * 50)], ((float*)x)[((incX * 48) + 1)], ((float*)x)[(incX * 48)]), abs_mask_tmp);
                    x_7 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 62) + 1)], ((float*)x)[(incX * 62)], ((float*)x)[((incX * 60) + 1)], ((float*)x)[(incX * 60)], ((float*)x)[((incX * 58) + 1)], ((float*)x)[(incX * 58)], ((float*)x)[((incX * 56) + 1)], ((float*)x)[(incX * 56)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_4 = _mm256_add_ps(x_4, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_4, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_5, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_5 = _mm256_add_ps(x_5, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_6 = _mm256_add_ps(x_6, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_6, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_7, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_7 = _mm256_add_ps(x_7, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_2 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_3 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_2 = _mm256_add_ps(x_2, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_2, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_3, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_3 = _mm256_add_ps(x_3, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_3, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_1, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_ps(_mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_ps((__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      x_0 = _mm256_add_ps(x_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = _mm256_sub_ps(s_buffer[j], _mm256_set_ps(((float*)asum)[((j * 2) + 1)], ((float*)asum)[(j * 2)], ((float*)asum)[((j * 2) + 1)], ((float*)asum)[(j * 2)], ((float*)asum)[((j * 2) + 1)], ((float*)asum)[(j * 2)], 0, 0));
                _mm256_store_ps(cons_buffer_tmp, s_buffer[j]);
                ((float*)asum)[(j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
                ((float*)asum)[((j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
        __m128 abs_mask_tmp;
        {
          __m128 tmp;
          tmp = _mm_set1_ps(1);
          abs_mask_tmp = _mm_set1_ps(-1);
          abs_mask_tmp = _mm_xor_ps(abs_mask_tmp, tmp);
          tmp = _mm_cmpeq_ps(tmp, tmp);
          abs_mask_tmp = _mm_xor_ps(abs_mask_tmp, tmp);
        }
        __m128 blp_mask_tmp;
        {
          __m128 tmp;
          blp_mask_tmp = _mm_set1_ps(1.0);
          tmp = _mm_set1_ps(1.0 + (FLT_EPSILON * 1.0001));
          blp_mask_tmp = _mm_xor_ps(blp_mask_tmp, tmp);
        }
        __m128 cons_tmp; (void)cons_tmp;
        float cons_buffer_tmp[4] __attribute__((aligned(16))); (void)cons_buffer_tmp;
        unsigned int SIMD_daz_ftz_old_tmp = 0;
        unsigned int SIMD_daz_ftz_new_tmp = 0;


        switch(fold){
          case 2:
            {
              int i;
              __m128 x_0, x_1, x_2, x_3;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;

              s_0_0 = s_0_1 = (__m128)_mm_load1_pd((double *)(((float*)asum)));
              s_1_0 = s_1_1 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 2));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 8 <= N_block; i += 8, x += 16){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 8 <= N_block; i += 8, x += 16){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)asum)[1], ((float*)asum)[0], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum))));
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)asum)[3], ((float*)asum)[2], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum)) + 2));
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128 x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;
              __m128 s_2_0, s_2_1;

              s_0_0 = s_0_1 = (__m128)_mm_load1_pd((double *)(((float*)asum)));
              s_1_0 = s_1_1 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 2));
              s_2_0 = s_2_1 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 4));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 20), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 28), abs_mask_tmp);
                    x_8 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_9 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 36), abs_mask_tmp);
                    x_10 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 40), abs_mask_tmp);
                    x_11 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 44), abs_mask_tmp);
                    x_12 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 48), abs_mask_tmp);
                    x_13 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 52), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(_mm_add_ps(x_6, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_7 = _mm_add_ps(_mm_add_ps(x_7, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_8 = _mm_add_ps(_mm_add_ps(x_8, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_9 = _mm_add_ps(_mm_add_ps(x_9, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_10 = _mm_add_ps(_mm_add_ps(x_10, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_11 = _mm_add_ps(_mm_add_ps(x_11, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_13, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_12 = _mm_add_ps(_mm_add_ps(x_12, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_13 = _mm_add_ps(_mm_add_ps(x_13, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 20), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(_mm_add_ps(x_6, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_7 = _mm_add_ps(_mm_add_ps(x_7, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 20), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 28), abs_mask_tmp);
                    x_8 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 32), abs_mask_tmp);
                    x_9 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 36), abs_mask_tmp);
                    x_10 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 40), abs_mask_tmp);
                    x_11 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 44), abs_mask_tmp);
                    x_12 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 48), abs_mask_tmp);
                    x_13 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 52), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 16), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 20), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 24), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)]), abs_mask_tmp);
                    x_12 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 50) + 1)], ((float*)x)[(incX * 50)], ((float*)x)[((incX * 48) + 1)], ((float*)x)[(incX * 48)]), abs_mask_tmp);
                    x_13 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 54) + 1)], ((float*)x)[(incX * 54)], ((float*)x)[((incX * 52) + 1)], ((float*)x)[(incX * 52)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(_mm_add_ps(x_6, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_7 = _mm_add_ps(_mm_add_ps(x_7, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_8 = _mm_add_ps(_mm_add_ps(x_8, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_9 = _mm_add_ps(_mm_add_ps(x_9, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_10 = _mm_add_ps(_mm_add_ps(x_10, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_11 = _mm_add_ps(_mm_add_ps(x_11, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_13, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_12 = _mm_add_ps(_mm_add_ps(x_12, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_13 = _mm_add_ps(_mm_add_ps(x_13, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(_mm_add_ps(x_6, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_7 = _mm_add_ps(_mm_add_ps(x_7, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 34) + 1)], ((float*)x)[(incX * 34)], ((float*)x)[((incX * 32) + 1)], ((float*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 38) + 1)], ((float*)x)[(incX * 38)], ((float*)x)[((incX * 36) + 1)], ((float*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 42) + 1)], ((float*)x)[(incX * 42)], ((float*)x)[((incX * 40) + 1)], ((float*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 46) + 1)], ((float*)x)[(incX * 46)], ((float*)x)[((incX * 44) + 1)], ((float*)x)[(incX * 44)]), abs_mask_tmp);
                    x_12 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 50) + 1)], ((float*)x)[(incX * 50)], ((float*)x)[((incX * 48) + 1)], ((float*)x)[(incX * 48)]), abs_mask_tmp);
                    x_13 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 54) + 1)], ((float*)x)[(incX * 54)], ((float*)x)[((incX * 52) + 1)], ((float*)x)[(incX * 52)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_8 = _mm_add_ps(x_8, q_0);
                    x_9 = _mm_add_ps(x_9, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_10 = _mm_add_ps(x_10, q_0);
                    x_11 = _mm_add_ps(x_11, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_12 = _mm_add_ps(x_12, q_0);
                    x_13 = _mm_add_ps(x_13, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_4 = _mm_add_ps(x_4, q_0);
                    x_5 = _mm_add_ps(x_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_6 = _mm_add_ps(x_6, q_0);
                    x_7 = _mm_add_ps(x_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_2 = _mm_add_ps(x_2, q_0);
                    x_3 = _mm_add_ps(x_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    x_0 = _mm_add_ps(x_0, q_0);
                    x_1 = _mm_add_ps(x_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)asum)[1], ((float*)asum)[0], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum))));
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)asum)[3], ((float*)asum)[2], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum)) + 2));
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(((float*)asum)[5], ((float*)asum)[4], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum)) + 4));
              s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128 x_0, x_1, x_2, x_3;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0;
              __m128 s_0_0;
              __m128 s_1_0;
              __m128 s_2_0;
              __m128 s_3_0;

              s_0_0 = (__m128)_mm_load1_pd((double *)(((float*)asum)));
              s_1_0 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 2));
              s_2_0 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 4));
              s_3_0 = (__m128)_mm_load1_pd((double *)(((float*)asum) + 6));

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 8 <= N_block; i += 8, x += 16){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 8 <= N_block; i += 8, x += 16){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 8), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_2 = _mm_add_ps(x_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_3 = _mm_add_ps(x_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_3, blp_mask_tmp));
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_1 = _mm_add_ps(x_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    x_0 = _mm_add_ps(x_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)asum)[1], ((float*)asum)[0], 0, 0));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)asum)[3], ((float*)asum)[2], 0, 0));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(((float*)asum)[5], ((float*)asum)[4], 0, 0));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_3_0 = _mm_sub_ps(s_3_0, _mm_set_ps(((float*)asum)[7], ((float*)asum)[6], 0, 0));
              _mm_store_ps(cons_buffer_tmp, s_3_0);
              ((float*)asum)[6] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)asum)[7] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128 x_0, x_1;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0, s_1;
              __m128 s_buffer[(binned_SBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = (__m128)_mm_load1_pd((double *)(((float*)asum) + (j * 2)));
              }

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 4 <= N_block; i += 4, x += 8){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    q_1 = _mm_sub_ps(s_1, q_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += 8){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_loadu_ps(((float*)x) + 4), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_loadu_ps(((float*)x)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                      expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_ps(1.0, binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION);
                      expansion_0 = _mm_set_ps(1.0, binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_ps(0.0, binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_ps(binned_SMCOMPRESSION, 1.0, binned_SMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 1.0, binned_SMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_ps(binned_SMEXPANSION * 0.5, 0.0, binned_SMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    q_1 = _mm_sub_ps(s_1, q_1);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_ps(s_0, q_0);
                    x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                    x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8)){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);
                    x_1 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_ps(s_buffer[((j * 2) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm_and_ps(_mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      x_0 = _mm_add_ps(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_or_ps(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = _mm_sub_ps(s_buffer[(j * 2)], _mm_set_ps(((float*)asum)[((j * 2) + 1)], ((float*)asum)[(j * 2)], 0, 0));
                cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)asum)) + (j * 2)));
                s_buffer[(j * 2)] = _mm_add_ps(s_buffer[(j * 2)], _mm_sub_ps(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm_store_ps(cons_buffer_tmp, s_buffer[(j * 2)]);
                ((float*)asum)[(j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
                ((float*)asum)[((j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #else
        int_float blp_tmp; (void)blp_tmp;
        float cons_tmp; (void)cons_tmp;


        switch(fold){
          case 3:
            {
              int i;
              float x_0, x_1;
              float compression_0, compression_1;
              float expansion_0, expansion_1;
              float expansion_mask_0, expansion_mask_1;
              float q_0, q_1;
              float s_0_0, s_0_1;
              float s_1_0, s_1_1;
              float s_2_0, s_2_1;

              s_0_0 = ((float*)asum)[0];
              s_0_1 = ((float*)asum)[1];
              s_1_0 = ((float*)asum)[2];
              s_1_1 = ((float*)asum)[3];
              s_2_0 = ((float*)asum)[4];
              s_2_1 = ((float*)asum)[5];

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = binned_SMCOMPRESSION;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = binned_SMEXPANSION * 0.5;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = binned_SMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_SMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_SMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_SMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.f = (x_0 * compression_0);
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    blp_tmp.f = (x_1 * compression_1);
                    blp_tmp.i |= 1;
                    s_0_1 = s_0_1 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_1_1 = s_1_1 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_2_1 = s_2_1 + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_0_1 = s_0_1 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_1_1 = s_1_1 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_2_1 = s_2_1 + blp_tmp.f;
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = binned_SMCOMPRESSION;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = binned_SMEXPANSION * 0.5;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = binned_SMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_SMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_SMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_SMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.f = (x_0 * compression_0);
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    blp_tmp.f = (x_1 * compression_1);
                    blp_tmp.i |= 1;
                    s_0_1 = s_0_1 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_1_1 = s_1_1 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_2_1 = s_2_1 + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_0_1 = s_0_1 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_1_1 = s_1_1 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_2_1 = s_2_1 + blp_tmp.f;
                  }
                }
              }

              ((float*)asum)[0] = s_0_0;
              ((float*)asum)[1] = s_0_1;
              ((float*)asum)[2] = s_1_0;
              ((float*)asum)[3] = s_1_1;
              ((float*)asum)[4] = s_2_0;
              ((float*)asum)[5] = s_2_1;

            }
            break;
          default:
            {
              int i, j;
              float x_0, x_1;
              float compression_0, compression_1;
              float expansion_0, expansion_1;
              float expansion_mask_0, expansion_mask_1;
              float q_0, q_1;
              float s_0, s_1;
              float s_buffer[(binned_SBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = ((float*)asum)[(j * 2)];
                s_buffer[((j * 2) + 1)] = ((float*)asum)[((j * 2) + 1)];
              }

              if(incX == 1){
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = binned_SMCOMPRESSION;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = binned_SMEXPANSION * 0.5;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = binned_SMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_SMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_SMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_SMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    blp_tmp.f = (x_0 * compression_0);
                    blp_tmp.i |= 1;
                    q_0 = s_0 + blp_tmp.f;
                    blp_tmp.f = (x_1 * compression_1);
                    blp_tmp.i |= 1;
                    q_1 = s_1 + blp_tmp.f;
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = (s_0 - q_0);
                    q_1 = (s_1 - q_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.f;
                  }
                }
              }else{
                if(binned_smindex0(asum) || binned_smindex0(asum + 1)){
                  if(binned_smindex0(asum)){
                    if(binned_smindex0(asum + 1)){
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = binned_SMCOMPRESSION;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = binned_SMEXPANSION * 0.5;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = binned_SMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_SMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_SMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_SMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_SMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_SMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_SMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    blp_tmp.f = (x_0 * compression_0);
                    blp_tmp.i |= 1;
                    q_0 = s_0 + blp_tmp.f;
                    blp_tmp.f = (x_1 * compression_1);
                    blp_tmp.i |= 1;
                    q_1 = s_1 + blp_tmp.f;
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = (s_0 - q_0);
                    q_1 = (s_1 - q_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabsf(((float*)x)[0]);
                    x_1 = fabsf(((float*)x)[1]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.f = x_0;
                    blp_tmp.i |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.f;
                    blp_tmp.f = x_1;
                    blp_tmp.i |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.f;
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                ((float*)asum)[(j * 2)] = s_buffer[(j * 2)];
                ((float*)asum)[((j * 2) + 1)] = s_buffer[((j * 2) + 1)];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    deposits += N_block;
  }

  binned_cbrenorm(fold, asum);
  binned_smsmadd(fold, asum, 2, asum + 2 * fold, 2, priY, incpriY, carY, inccarY);
  binned_smsmadd(fold, asum + 1, 2, asum + 2 * fold + 1, 2, priY, incpriY, carY, inccarY);

  free(asum);

  return;
}