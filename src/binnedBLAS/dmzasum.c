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

cog.out(generate.generate(blockSize.BlockSize("dmzasum", "N_block_MAX", 32, terminal.get_diendurance()//2, terminal.get_diendurance()//2, ["bench_rdzasum_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
]]]*/
#if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
  #include <immintrin.h>

#elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
  #include <emmintrin.h>

#else


#endif

#define N_block_MAX 1024
//[[[end]]]

/**
 * @internal
 * @brief Add to manually specified binned double precision Y the absolute sum of complex double precision vector X
 *
 * Add to Y the binned sum of magnitudes of elements of X.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X complex double precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param priY Y's primary vector
 * @param incpriY stride within Y's primary vector (use every incpriY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_dmzasum(const int fold, const int N, const void *X, const int incX, double *priY, const int incpriY, double *carY, const int inccarY){
  double amax_tmp[2];
  double amax;
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;
  double_complex_binned *asum = binned_zballoc(fold);
  binned_zbsetzero(fold, asum);

  const double *x = (const double*)X;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    binnedBLAS_zamax_sub(N_block, x, incX, amax_tmp);
    amax = MAX(amax_tmp[0], amax_tmp[1]);

    if (isinf(amax) || isinf(priY[0])){
      for (j = 0; j < N_block; j++){
        priY[0] += fabs(x[j * 2 * incX]);
        priY[0] += fabs(x[j * 2 * incX + 1]);
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

    if (deposits + N_block > binned_DBENDURANCE) {
      binned_zbrenorm(fold, asum);
      deposits = 0;
    }

    binned_zbdupdate(fold, amax, asum);

    /*[[[cog
      cog.out(generate.generate(depositASum.DepositASum(dataTypes.DoubleComplex, "fold", "N_block", "x", "incX", "asum", 1), cog.inFile, args, params, mode))
      ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
        __m256d abs_mask_tmp;
        {
          __m256d tmp;
          tmp = _mm256_set1_pd(1);
          abs_mask_tmp = _mm256_set1_pd(-1);
          abs_mask_tmp = _mm256_xor_pd(abs_mask_tmp, tmp);
          tmp = _mm256_cmp_pd(tmp, tmp, 0);
          abs_mask_tmp = _mm256_xor_pd(abs_mask_tmp, tmp);
        }
        __m256d blp_mask_tmp;
        {
          __m256d tmp;
          blp_mask_tmp = _mm256_set1_pd(1.0);
          tmp = _mm256_set1_pd(1.0 + (DBL_EPSILON * 1.0001));
          blp_mask_tmp = _mm256_xor_pd(blp_mask_tmp, tmp);
        }
        __m256d cons_tmp; (void)cons_tmp;
        double cons_buffer_tmp[4] __attribute__((aligned(32))); (void)cons_buffer_tmp;
        unsigned int SIMD_daz_ftz_old_tmp = 0;
        unsigned int SIMD_daz_ftz_new_tmp = 0;


        switch(fold){
          case 2:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 2));

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 44), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 44), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)asum)[1], ((double*)asum)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)asum)[3], ((double*)asum)[2], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 2));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 2));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 4));

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)asum)[1], ((double*)asum)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)asum)[3], ((double*)asum)[2], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 2));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)asum)[5], ((double*)asum)[4], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 4));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;
              __m256d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 2));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 4));
              s_3_0 = s_3_1 = _mm256_broadcast_pd((__m128d *)(((double*)asum) + 6));

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_6 = _mm256_add_pd(x_6, q_0);
                    x_7 = _mm256_add_pd(x_7, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_2 = _mm256_add_pd(x_2, q_0);
                    x_3 = _mm256_add_pd(x_3, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)asum)[1], ((double*)asum)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)asum)[3], ((double*)asum)[2], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 2));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[3] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)asum)[5], ((double*)asum)[4], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 4));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)asum)[4] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[5] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_3_0 = _mm256_sub_pd(s_3_0, _mm256_set_pd(((double*)asum)[7], ((double*)asum)[6], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + 6));
              s_3_0 = _mm256_add_pd(s_3_0, _mm256_sub_pd(s_3_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)asum)[6] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)asum)[7] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0, s_1;
              __m256d s_buffer[(binned_DBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm256_broadcast_pd((__m128d *)(((double*)asum) + (j * 2)));
              }

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 44), abs_mask_tmp);
                    x_12 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 48), abs_mask_tmp);
                    x_13 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 52), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_10 = _mm256_add_pd(x_10, q_0);
                      x_11 = _mm256_add_pd(x_11, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_11, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_12, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_12 = _mm256_add_pd(_mm256_add_pd(x_12, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm256_add_pd(_mm256_add_pd(x_13, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_12, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_13, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_12 = _mm256_add_pd(x_12, q_0);
                      x_13 = _mm256_add_pd(x_13, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_12, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 44), abs_mask_tmp);
                    x_12 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 48), abs_mask_tmp);
                    x_13 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 52), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_10 = _mm256_add_pd(x_10, q_0);
                      x_11 = _mm256_add_pd(x_11, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_11, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_12, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_13, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_12 = _mm256_add_pd(x_12, q_0);
                      x_13 = _mm256_add_pd(x_13, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_12, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 28), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 12), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_loadu_pd(((double*)x) + 4), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_loadu_pd(((double*)x)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm256_set_pd(1.0, binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm256_set_pd(1.0, binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm256_set_pd(0.0, binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm256_set_pd(binned_DMCOMPRESSION, 1.0, binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 1.0, binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm256_set_pd(binned_DMEXPANSION * 0.5, 0.0, binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]), abs_mask_tmp);
                    x_12 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 50) + 1)], ((double*)x)[(incX * 50)], ((double*)x)[((incX * 48) + 1)], ((double*)x)[(incX * 48)]), abs_mask_tmp);
                    x_13 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 54) + 1)], ((double*)x)[(incX * 54)], ((double*)x)[((incX * 52) + 1)], ((double*)x)[(incX * 52)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_10 = _mm256_add_pd(x_10, q_0);
                      x_11 = _mm256_add_pd(x_11, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_11, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_12, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_12 = _mm256_add_pd(_mm256_add_pd(x_12, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm256_add_pd(_mm256_add_pd(x_13, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_12, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_13, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_12 = _mm256_add_pd(x_12, q_0);
                      x_13 = _mm256_add_pd(x_13, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_12, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);
                    x_8 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]), abs_mask_tmp);
                    x_9 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]), abs_mask_tmp);
                    x_10 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]), abs_mask_tmp);
                    x_11 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]), abs_mask_tmp);
                    x_12 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 50) + 1)], ((double*)x)[(incX * 50)], ((double*)x)[((incX * 48) + 1)], ((double*)x)[(incX * 48)]), abs_mask_tmp);
                    x_13 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 54) + 1)], ((double*)x)[(incX * 54)], ((double*)x)[((incX * 52) + 1)], ((double*)x)[(incX * 52)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_10 = _mm256_add_pd(x_10, q_0);
                      x_11 = _mm256_add_pd(x_11, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_11, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_12, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_13, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_12 = _mm256_add_pd(x_12, q_0);
                      x_13 = _mm256_add_pd(x_13, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_12, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_13, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);
                    x_4 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]), abs_mask_tmp);
                    x_5 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]), abs_mask_tmp);
                    x_6 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]), abs_mask_tmp);
                    x_7 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_6 = _mm256_add_pd(x_6, q_0);
                      x_7 = _mm256_add_pd(x_7, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);
                    x_2 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]), abs_mask_tmp);
                    x_3 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_0);
                      x_3 = _mm256_add_pd(x_3, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);
                    x_1 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_and_pd(_mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = _mm256_sub_pd(s_buffer[(j * 2)], _mm256_set_pd(((double*)asum)[((j * 2) + 1)], ((double*)asum)[(j * 2)], 0, 0));
                cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)asum)) + (j * 2)));
                s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_sub_pd(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm256_store_pd(cons_buffer_tmp, s_buffer[(j * 2)]);
                ((double*)asum)[(j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
                ((double*)asum)[((j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
        __m128d abs_mask_tmp;
        {
          __m128d tmp;
          tmp = _mm_set1_pd(1);
          abs_mask_tmp = _mm_set1_pd(-1);
          abs_mask_tmp = _mm_xor_pd(abs_mask_tmp, tmp);
          tmp = _mm_cmpeq_pd(tmp, tmp);
          abs_mask_tmp = _mm_xor_pd(abs_mask_tmp, tmp);
        }
        __m128d blp_mask_tmp;
        {
          __m128d tmp;
          blp_mask_tmp = _mm_set1_pd(1.0);
          tmp = _mm_set1_pd(1.0 + (DBL_EPSILON * 1.0001));
          blp_mask_tmp = _mm_xor_pd(blp_mask_tmp, tmp);
        }
        __m128d cons_tmp; (void)cons_tmp;
        double cons_buffer_tmp[2] __attribute__((aligned(16))); (void)cons_buffer_tmp;
        unsigned int SIMD_daz_ftz_old_tmp = 0;
        unsigned int SIMD_daz_ftz_new_tmp = 0;


        switch(fold){
          case 2:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1, q_2, q_3;
              __m128d s_0_0, s_0_1, s_0_2, s_0_3;
              __m128d s_1_0, s_1_1, s_1_2, s_1_3;

              s_0_0 = s_0_1 = s_0_2 = s_0_3 = _mm_loadu_pd(((double*)asum));
              s_1_0 = s_1_1 = s_1_2 = s_1_3 = _mm_loadu_pd(((double*)asum) + 2);

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);
                    x_16 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_17 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 34), abs_mask_tmp);
                    x_18 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_19 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 38), abs_mask_tmp);
                    x_20 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_21 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 42), abs_mask_tmp);
                    x_22 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 44), abs_mask_tmp);
                    x_23 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 46), abs_mask_tmp);
                    x_24 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 48), abs_mask_tmp);
                    x_25 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 50), abs_mask_tmp);
                    x_26 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 52), abs_mask_tmp);
                    x_27 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 54), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_16, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_17, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_18, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_19, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_16 = _mm_add_pd(_mm_add_pd(x_16, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_17 = _mm_add_pd(_mm_add_pd(x_17, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_18 = _mm_add_pd(_mm_add_pd(x_18, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_19 = _mm_add_pd(_mm_add_pd(x_19, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_20, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_21, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_22, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_23, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_20 = _mm_add_pd(_mm_add_pd(x_20, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_21 = _mm_add_pd(_mm_add_pd(x_21, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_22 = _mm_add_pd(_mm_add_pd(x_22, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_23 = _mm_add_pd(_mm_add_pd(x_23, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_24, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_25, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_26, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_27, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_24 = _mm_add_pd(_mm_add_pd(x_24, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_25 = _mm_add_pd(_mm_add_pd(x_25, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_26 = _mm_add_pd(_mm_add_pd(x_26, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_27 = _mm_add_pd(_mm_add_pd(x_27, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_27, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += 56){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);
                    x_16 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 32), abs_mask_tmp);
                    x_17 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 34), abs_mask_tmp);
                    x_18 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 36), abs_mask_tmp);
                    x_19 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 38), abs_mask_tmp);
                    x_20 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 40), abs_mask_tmp);
                    x_21 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 42), abs_mask_tmp);
                    x_22 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 44), abs_mask_tmp);
                    x_23 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 46), abs_mask_tmp);
                    x_24 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 48), abs_mask_tmp);
                    x_25 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 50), abs_mask_tmp);
                    x_26 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 52), abs_mask_tmp);
                    x_27 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 54), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    x_10 = _mm_add_pd(x_10, q_2);
                    x_11 = _mm_add_pd(x_11, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    x_14 = _mm_add_pd(x_14, q_2);
                    x_15 = _mm_add_pd(x_15, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_16 = _mm_add_pd(x_16, q_0);
                    x_17 = _mm_add_pd(x_17, q_1);
                    x_18 = _mm_add_pd(x_18, q_2);
                    x_19 = _mm_add_pd(x_19, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_20 = _mm_add_pd(x_20, q_0);
                    x_21 = _mm_add_pd(x_21, q_1);
                    x_22 = _mm_add_pd(x_22, q_2);
                    x_23 = _mm_add_pd(x_23, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_27, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_24 = _mm_add_pd(x_24, q_0);
                    x_25 = _mm_add_pd(x_25, q_1);
                    x_26 = _mm_add_pd(x_26, q_2);
                    x_27 = _mm_add_pd(x_27, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_27, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    x_10 = _mm_add_pd(x_10, q_2);
                    x_11 = _mm_add_pd(x_11, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    x_14 = _mm_add_pd(x_14, q_2);
                    x_15 = _mm_add_pd(x_15, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);
                    x_16 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 32)), abs_mask_tmp);
                    x_17 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 34)), abs_mask_tmp);
                    x_18 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 36)), abs_mask_tmp);
                    x_19 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 38)), abs_mask_tmp);
                    x_20 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 40)), abs_mask_tmp);
                    x_21 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 42)), abs_mask_tmp);
                    x_22 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 44)), abs_mask_tmp);
                    x_23 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 46)), abs_mask_tmp);
                    x_24 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 48)), abs_mask_tmp);
                    x_25 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 50)), abs_mask_tmp);
                    x_26 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 52)), abs_mask_tmp);
                    x_27 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 54)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_16, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_17, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_18, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_19, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_16 = _mm_add_pd(_mm_add_pd(x_16, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_17 = _mm_add_pd(_mm_add_pd(x_17, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_18 = _mm_add_pd(_mm_add_pd(x_18, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_19 = _mm_add_pd(_mm_add_pd(x_19, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_20, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_21, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_22, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_23, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_20 = _mm_add_pd(_mm_add_pd(x_20, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_21 = _mm_add_pd(_mm_add_pd(x_21, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_22 = _mm_add_pd(_mm_add_pd(x_22, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_23 = _mm_add_pd(_mm_add_pd(x_23, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_24, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_25, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_26, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_27, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_24 = _mm_add_pd(_mm_add_pd(x_24, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_25 = _mm_add_pd(_mm_add_pd(x_25, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_26 = _mm_add_pd(_mm_add_pd(x_26, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_27 = _mm_add_pd(_mm_add_pd(x_27, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_27, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, x += (incX * 56)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);
                    x_16 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 32)), abs_mask_tmp);
                    x_17 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 34)), abs_mask_tmp);
                    x_18 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 36)), abs_mask_tmp);
                    x_19 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 38)), abs_mask_tmp);
                    x_20 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 40)), abs_mask_tmp);
                    x_21 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 42)), abs_mask_tmp);
                    x_22 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 44)), abs_mask_tmp);
                    x_23 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 46)), abs_mask_tmp);
                    x_24 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 48)), abs_mask_tmp);
                    x_25 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 50)), abs_mask_tmp);
                    x_26 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 52)), abs_mask_tmp);
                    x_27 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 54)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    x_10 = _mm_add_pd(x_10, q_2);
                    x_11 = _mm_add_pd(x_11, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    x_14 = _mm_add_pd(x_14, q_2);
                    x_15 = _mm_add_pd(x_15, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_16 = _mm_add_pd(x_16, q_0);
                    x_17 = _mm_add_pd(x_17, q_1);
                    x_18 = _mm_add_pd(x_18, q_2);
                    x_19 = _mm_add_pd(x_19, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_16, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_17, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_18, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_19, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_20 = _mm_add_pd(x_20, q_0);
                    x_21 = _mm_add_pd(x_21, q_1);
                    x_22 = _mm_add_pd(x_22, q_2);
                    x_23 = _mm_add_pd(x_23, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_20, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_21, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_22, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_23, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_27, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_24 = _mm_add_pd(x_24, q_0);
                    x_25 = _mm_add_pd(x_25, q_1);
                    x_26 = _mm_add_pd(x_26, q_2);
                    x_27 = _mm_add_pd(x_27, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_24, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_25, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_26, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_27, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    x_10 = _mm_add_pd(x_10, q_2);
                    x_11 = _mm_add_pd(x_11, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    x_14 = _mm_add_pd(x_14, q_2);
                    x_15 = _mm_add_pd(x_15, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_15, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    x_6 = _mm_add_pd(x_6, q_2);
                    x_7 = _mm_add_pd(x_7, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    q_2 = _mm_sub_pd(q_2, s_0_2);
                    q_3 = _mm_sub_pd(q_3, s_0_3);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    x_2 = _mm_add_pd(x_2, q_2);
                    x_3 = _mm_add_pd(x_3, q_3);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_2, cons_tmp));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_3, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0];
              ((double*)asum)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 2);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_2, cons_tmp));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_3, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0];
              ((double*)asum)[3] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)asum));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)asum) + 2);
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)asum) + 4);

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 12 <= N_block; i += 12, x += 24){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 12 <= N_block; i += 12, x += 24){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 12 <= N_block; i += 12, x += (incX * 24)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 12 <= N_block; i += 12, x += (incX * 24)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0];
              ((double*)asum)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 2);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0];
              ((double*)asum)[3] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 4);
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)asum)[4] = cons_buffer_tmp[0];
              ((double*)asum)[5] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;
              __m128d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)asum));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)asum) + 2);
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)asum) + 4);
              s_3_0 = s_3_1 = _mm_loadu_pd(((double*)asum) + 6);

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_15, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 16), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 18), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 20), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 22), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 24), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 26), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 28), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 30), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_15, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 8), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 10), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 12), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 14), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(_mm_add_pd(x_8, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm_add_pd(_mm_add_pd(x_9, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(_mm_add_pd(x_10, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm_add_pd(_mm_add_pd(x_11, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_12, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_13, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_12 = _mm_add_pd(_mm_add_pd(x_12, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_13 = _mm_add_pd(_mm_add_pd(x_13, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_14, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_15, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_14 = _mm_add_pd(_mm_add_pd(x_14, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_15 = _mm_add_pd(_mm_add_pd(x_15, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_15, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);
                    x_8 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 16)), abs_mask_tmp);
                    x_9 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 18)), abs_mask_tmp);
                    x_10 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 20)), abs_mask_tmp);
                    x_11 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 22)), abs_mask_tmp);
                    x_12 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 24)), abs_mask_tmp);
                    x_13 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 26)), abs_mask_tmp);
                    x_14 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 28)), abs_mask_tmp);
                    x_15 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 30)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_8 = _mm_add_pd(x_8, q_0);
                    x_9 = _mm_add_pd(x_9, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_10 = _mm_add_pd(x_10, q_0);
                    x_11 = _mm_add_pd(x_11, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_11, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_12 = _mm_add_pd(x_12, q_0);
                    x_13 = _mm_add_pd(x_13, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_12, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_13, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_15, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_14 = _mm_add_pd(x_14, q_0);
                    x_15 = _mm_add_pd(x_15, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_14, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_15, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);
                    x_4 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 8)), abs_mask_tmp);
                    x_5 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 10)), abs_mask_tmp);
                    x_6 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 12)), abs_mask_tmp);
                    x_7 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 14)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_4 = _mm_add_pd(x_4, q_0);
                    x_5 = _mm_add_pd(x_5, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_4, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_6 = _mm_add_pd(x_6, q_0);
                    x_7 = _mm_add_pd(x_7, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_6, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_7, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_3, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    q_1 = _mm_sub_pd(q_1, s_1_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(x_1, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    q_1 = _mm_sub_pd(q_1, s_2_1);
                    x_0 = _mm_add_pd(x_0, q_0);
                    x_1 = _mm_add_pd(x_1, q_1);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_1_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_2_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)asum)[0] = cons_buffer_tmp[0];
              ((double*)asum)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 2);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)asum)[2] = cons_buffer_tmp[0];
              ((double*)asum)[3] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 4);
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)asum)[4] = cons_buffer_tmp[0];
              ((double*)asum)[5] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + 6);
              s_3_0 = _mm_add_pd(s_3_0, _mm_sub_pd(s_3_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)asum)[6] = cons_buffer_tmp[0];
              ((double*)asum)[7] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128d x_0, x_1, x_2, x_3;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1, q_2, q_3;
              __m128d s_0, s_1, s_2, s_3;
              __m128d s_buffer[(binned_DBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = _mm_loadu_pd(((double*)asum) + (j * 2));
              }

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 4 <= N_block; i += 4, x += 8){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    q_1 = _mm_sub_pd(s_1, q_1);
                    q_2 = _mm_sub_pd(s_2, q_2);
                    q_3 = _mm_sub_pd(s_3, q_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                      x_2 = _mm_add_pd(x_2, q_2);
                      x_3 = _mm_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    q_1 = _mm_sub_pd(s_1, q_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += 8){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 4), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 6), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                      x_2 = _mm_add_pd(x_2, q_2);
                      x_3 = _mm_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + 2), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                      expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    }else{
                      compression_0 = _mm_set_pd(1.0, binned_DMCOMPRESSION);
                      expansion_0 = _mm_set_pd(1.0, binned_DMEXPANSION * 0.5);
                      expansion_mask_0 = _mm_set_pd(0.0, binned_DMEXPANSION * 0.5);
                    }
                  }else{
                    compression_0 = _mm_set_pd(binned_DMCOMPRESSION, 1.0);
                    expansion_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 1.0);
                    expansion_mask_0 = _mm_set_pd(binned_DMEXPANSION * 0.5, 0.0);
                  }
                  for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    q_1 = _mm_sub_pd(s_1, q_1);
                    q_2 = _mm_sub_pd(s_2, q_2);
                    q_3 = _mm_sub_pd(s_3, q_3);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                      x_2 = _mm_add_pd(x_2, q_2);
                      x_3 = _mm_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    q_1 = _mm_sub_pd(s_1, q_1);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8)){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);
                    x_2 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 4)), abs_mask_tmp);
                    x_3 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 6)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                      x_2 = _mm_add_pd(x_2, q_2);
                      x_3 = _mm_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);
                    x_1 = _mm_and_pd(_mm_loadu_pd(((double*)x) + (incX * 2)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_and_pd(_mm_loadu_pd(((double*)x)), abs_mask_tmp);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                cons_tmp = _mm_loadu_pd(((double*)((double*)asum)) + (j * 2));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 1)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 2)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 3)], cons_tmp));
                _mm_store_pd(cons_buffer_tmp, s_buffer[(j * 4)]);
                ((double*)asum)[(j * 2)] = cons_buffer_tmp[0];
                ((double*)asum)[((j * 2) + 1)] = cons_buffer_tmp[1];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #else
        long_double blp_tmp; (void)blp_tmp;
        double cons_tmp; (void)cons_tmp;


        switch(fold){
          case 3:
            {
              int i;
              double x_0, x_1;
              double compression_0, compression_1;
              double expansion_0, expansion_1;
              double expansion_mask_0, expansion_mask_1;
              double q_0, q_1;
              double s_0_0, s_0_1;
              double s_1_0, s_1_1;
              double s_2_0, s_2_1;

              s_0_0 = ((double*)asum)[0];
              s_0_1 = ((double*)asum)[1];
              s_1_0 = ((double*)asum)[2];
              s_1_1 = ((double*)asum)[3];
              s_2_0 = ((double*)asum)[4];
              s_2_1 = ((double*)asum)[5];

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = binned_DMCOMPRESSION;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = binned_DMEXPANSION * 0.5;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = binned_DMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_DMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_DMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_DMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.d = (x_0 * compression_0);
                    blp_tmp.l |= 1;
                    s_0_0 = s_0_0 + blp_tmp.d;
                    blp_tmp.d = (x_1 * compression_1);
                    blp_tmp.l |= 1;
                    s_0_1 = s_0_1 + blp_tmp.d;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_1_0 = s_1_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_1_1 = s_1_1 + blp_tmp.d;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_2_0 = s_2_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_2_1 = s_2_1 + blp_tmp.d;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_0_0 = s_0_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_0_1 = s_0_1 + blp_tmp.d;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_1_0 = s_1_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_1_1 = s_1_1 + blp_tmp.d;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_2_0 = s_2_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_2_1 = s_2_1 + blp_tmp.d;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = binned_DMCOMPRESSION;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = binned_DMEXPANSION * 0.5;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = binned_DMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_DMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_DMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_DMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.d = (x_0 * compression_0);
                    blp_tmp.l |= 1;
                    s_0_0 = s_0_0 + blp_tmp.d;
                    blp_tmp.d = (x_1 * compression_1);
                    blp_tmp.l |= 1;
                    s_0_1 = s_0_1 + blp_tmp.d;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_1_0 = s_1_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_1_1 = s_1_1 + blp_tmp.d;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_2_0 = s_2_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_2_1 = s_2_1 + blp_tmp.d;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_0_0 = s_0_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_0_1 = s_0_1 + blp_tmp.d;
                    q_0 = (q_0 - s_0_0);
                    q_1 = (q_1 - s_0_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_1_0 = s_1_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_1_1 = s_1_1 + blp_tmp.d;
                    q_0 = (q_0 - s_1_0);
                    q_1 = (q_1 - s_1_1);
                    x_0 = (x_0 + q_0);
                    x_1 = (x_1 + q_1);
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_2_0 = s_2_0 + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_2_1 = s_2_1 + blp_tmp.d;
                  }
                }
              }

              ((double*)asum)[0] = s_0_0;
              ((double*)asum)[1] = s_0_1;
              ((double*)asum)[2] = s_1_0;
              ((double*)asum)[3] = s_1_1;
              ((double*)asum)[4] = s_2_0;
              ((double*)asum)[5] = s_2_1;

            }
            break;
          default:
            {
              int i, j;
              double x_0, x_1;
              double compression_0, compression_1;
              double expansion_0, expansion_1;
              double expansion_mask_0, expansion_mask_1;
              double q_0, q_1;
              double s_0, s_1;
              double s_buffer[(binned_DBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = ((double*)asum)[(j * 2)];
                s_buffer[((j * 2) + 1)] = ((double*)asum)[((j * 2) + 1)];
              }

              if(incX == 1){
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = binned_DMCOMPRESSION;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = binned_DMEXPANSION * 0.5;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = binned_DMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_DMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_DMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_DMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    blp_tmp.d = (x_0 * compression_0);
                    blp_tmp.l |= 1;
                    q_0 = s_0 + blp_tmp.d;
                    blp_tmp.d = (x_1 * compression_1);
                    blp_tmp.l |= 1;
                    q_1 = s_1 + blp_tmp.d;
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = (s_0 - q_0);
                    q_1 = (s_1 - q_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.d;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += 2){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.d;
                  }
                }
              }else{
                if(binned_dmindex0(asum) || binned_dmindex0(asum + 1)){
                  if(binned_dmindex0(asum)){
                    if(binned_dmindex0(asum + 1)){
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = binned_DMCOMPRESSION;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = binned_DMEXPANSION * 0.5;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = binned_DMEXPANSION * 0.5;
                    }else{
                      compression_0 = binned_DMCOMPRESSION;
                      compression_1 = 1.0;
                      expansion_0 = binned_DMEXPANSION * 0.5;
                      expansion_1 = 1.0;
                      expansion_mask_0 = binned_DMEXPANSION * 0.5;
                      expansion_mask_1 = 0.0;
                    }
                  }else{
                    compression_0 = 1.0;
                    compression_1 = binned_DMCOMPRESSION;
                    expansion_0 = 1.0;
                    expansion_1 = binned_DMEXPANSION * 0.5;
                    expansion_mask_0 = 0.0;
                    expansion_mask_1 = binned_DMEXPANSION * 0.5;
                  }
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    blp_tmp.d = (x_0 * compression_0);
                    blp_tmp.l |= 1;
                    q_0 = s_0 + blp_tmp.d;
                    blp_tmp.d = (x_1 * compression_1);
                    blp_tmp.l |= 1;
                    q_1 = s_1 + blp_tmp.d;
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = (s_0 - q_0);
                    q_1 = (s_1 - q_1);
                    x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                    x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.d;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2)){
                    x_0 = fabs(((double*)x)[0]);
                    x_1 = fabs(((double*)x)[1]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                    }
                    blp_tmp.d = x_0;
                    blp_tmp.l |= 1;
                    s_buffer[(j * 2)] = s_buffer[(j * 2)] + blp_tmp.d;
                    blp_tmp.d = x_1;
                    blp_tmp.l |= 1;
                    s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + blp_tmp.d;
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                ((double*)asum)[(j * 2)] = s_buffer[(j * 2)];
                ((double*)asum)[((j * 2) + 1)] = s_buffer[((j * 2) + 1)];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    deposits += N_block;
  }

  binned_zbrenorm(fold, asum);
  binned_dmdmadd(fold, asum, 2, asum + 2 * fold, 2, priY, incpriY, carY, inccarY);
  binned_dmdmadd(fold, asum + 1, 2, asum + 2 * fold + 1, 2, priY, incpriY, carY, inccarY);

  free(asum);

  return;
}