#include <stdlib.h>
#include <math.h>

#include "../config.h"
#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
import depositSum
import vectorizations
from src.common import blockSize
from scripts import terminal

code_block = generate.CodeBlock()
vectorizations.conditionally_include_vectorizations(code_block)
cog.out(str(code_block))

cog.outl()

cog.out(generate.generate(blockSize.BlockSize("zmzsum", "N_block_MAX", 32, terminal.get_diendurance(), terminal.get_diendurance(), ["bench_rzsum_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
]]]*/
#if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
  #include <immintrin.h>

#elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
  #include <emmintrin.h>

#else


#endif

#define N_block_MAX 2048
//[[[end]]]

/**
 * @internal
 * @brief Add to manually specified binned complex double precision Y the sum of complex double precision vector X
 *
 * Add to Y the binned sum of X.
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
void binnedBLAS_zmzsum(const int fold, const int N, const void *X, const int incX, double *priY, const int incpriY, double *carY, const int inccarY){
  double amax[2];
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;

  const double *x = (const double*)X;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    binnedBLAS_zamax_sub(N_block, x, incX, amax);

    if (isinf(amax[0]) || isinf(priY[0])){
      for (j = 0; j < N_block; j++){
        priY[0] += x[j * 2 * incX];
      }
    }
    if (isinf(amax[1]) || isinf(priY[1])){
      for (j = 0; j < N_block; j++){
        priY[1] += x[j * 2 * incX + 1];
      }
    }
    if (isnan(priY[0]) && isnan(priY[1])){
      return;
    } else if (isinf(priY[0]) && isinf(priY[1])){
      x += N_block * 2 * incX;
      continue;
    }
    if (ISNANINF(priY[0])){
      amax[0] = priY[0];
    }
    if (ISNANINF(priY[1])){
      amax[1] = priY[1];
    }

    if (deposits + N_block > binned_DBENDURANCE) {
      binned_zmrenorm(fold, priY, incpriY, carY, inccarY);
      deposits = 0;
    }

    binned_zmzupdate(fold, amax, priY, incpriY, carY, inccarY);

    /*[[[cog
    cog.out(generate.generate(depositSum.DepositSum(dataTypes.DoubleComplex, "fold", "N_block", "x", "incX", "priY", "incpriY"), cog.inFile, args, params, mode))
    ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
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
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1, q_2, q_3;
              __m256d s_0_0, s_0_1, s_0_2, s_0_3;
              __m256d s_1_0, s_1_1, s_1_2, s_1_3;

              s_0_0 = s_0_1 = s_0_2 = s_0_3 = _mm256_broadcast_pd((__m128d *)(((double*)priY)));
              s_1_0 = s_1_1 = s_1_2 = s_1_3 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 2)));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += 32){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    x_2 = _mm256_add_pd(x_2, q_2);
                    x_3 = _mm256_add_pd(x_3, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    x_6 = _mm256_add_pd(x_6, q_2);
                    x_7 = _mm256_add_pd(x_7, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    x_2 = _mm256_add_pd(x_2, q_2);
                    x_3 = _mm256_add_pd(x_3, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32)){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    x_2 = _mm256_add_pd(x_2, q_2);
                    x_3 = _mm256_add_pd(x_3, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_7, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_4 = _mm256_add_pd(x_4, q_0);
                    x_5 = _mm256_add_pd(x_5, q_1);
                    x_6 = _mm256_add_pd(x_6, q_2);
                    x_7 = _mm256_add_pd(x_7, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_5, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_6, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_7, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    q_2 = s_0_2;
                    q_3 = s_0_3;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_0_2 = _mm256_add_pd(s_0_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_0_3 = _mm256_add_pd(s_0_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    q_2 = _mm256_sub_pd(q_2, s_0_2);
                    q_3 = _mm256_sub_pd(q_3, s_0_3);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    x_1 = _mm256_add_pd(x_1, q_1);
                    x_2 = _mm256_add_pd(x_2, q_2);
                    x_3 = _mm256_add_pd(x_3, q_3);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_1, blp_mask_tmp));
                    s_1_2 = _mm256_add_pd(s_1_2, _mm256_or_pd(x_2, blp_mask_tmp));
                    s_1_3 = _mm256_add_pd(s_1_3, _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    x_0 = _mm256_add_pd(x_0, q_0);
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priY)[1], ((double*)priY)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_2, cons_tmp));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_3, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priY)[((incpriY * 2) + 1)], ((double*)priY)[(incpriY * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_2, cons_tmp));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_3, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

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

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 2)));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 4)));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priY)[1], ((double*)priY)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priY)[((incpriY * 2) + 1)], ((double*)priY)[(incpriY * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)priY)[((incpriY * 4) + 1)], ((double*)priY)[(incpriY * 4)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 4)));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priY)[(incpriY * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;
              __m256d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 2)));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 4)));
              s_3_0 = s_3_1 = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * 6)));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);
                    x_8 = _mm256_loadu_pd(((double*)x) + 32);
                    x_9 = _mm256_loadu_pd(((double*)x) + 36);
                    x_10 = _mm256_loadu_pd(((double*)x) + 40);
                    x_11 = _mm256_loadu_pd(((double*)x) + 44);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

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
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);
                    x_8 = _mm256_loadu_pd(((double*)x) + 32);
                    x_9 = _mm256_loadu_pd(((double*)x) + 36);
                    x_10 = _mm256_loadu_pd(((double*)x) + 40);
                    x_11 = _mm256_loadu_pd(((double*)x) + 44);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

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
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_0 = _mm256_loadu_pd(((double*)x));

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);
                    x_8 = _mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]);
                    x_9 = _mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]);
                    x_10 = _mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]);
                    x_11 = _mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

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
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);
                    x_8 = _mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]);
                    x_9 = _mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]);
                    x_10 = _mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]);
                    x_11 = _mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_8 = _mm256_add_pd(x_8, q_0);
                    x_9 = _mm256_add_pd(x_9, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_8, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_9, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_0_0);
                    q_1 = _mm256_sub_pd(q_1, s_0_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_1_0);
                    q_1 = _mm256_sub_pd(q_1, s_1_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    q_0 = s_2_0;
                    q_1 = s_2_1;
                    s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(x_11, blp_mask_tmp));
                    q_0 = _mm256_sub_pd(q_0, s_2_0);
                    q_1 = _mm256_sub_pd(q_1, s_2_1);
                    x_10 = _mm256_add_pd(x_10, q_0);
                    x_11 = _mm256_add_pd(x_11, q_1);
                    s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(x_10, blp_mask_tmp));
                    s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

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
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

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
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

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

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priY)[1], ((double*)priY)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priY)[((incpriY * 2) + 1)], ((double*)priY)[(incpriY * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)priY)[((incpriY * 4) + 1)], ((double*)priY)[(incpriY * 4)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 4)));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priY)[(incpriY * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_3_0 = _mm256_sub_pd(s_3_0, _mm256_set_pd(((double*)priY)[((incpriY * 6) + 1)], ((double*)priY)[(incpriY * 6)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * 6)));
              s_3_0 = _mm256_add_pd(s_3_0, _mm256_sub_pd(s_3_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)priY)[(incpriY * 6)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priY)[((incpriY * 6) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1, q_2, q_3;
              __m256d s_0, s_1, s_2, s_3;
              __m256d s_buffer[(binned_DBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = _mm256_broadcast_pd((__m128d *)(((double*)priY) + (incpriY * j * 2)));
              }

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);
                    x_8 = _mm256_loadu_pd(((double*)x) + 32);
                    x_9 = _mm256_loadu_pd(((double*)x) + 36);
                    x_10 = _mm256_loadu_pd(((double*)x) + 40);
                    x_11 = _mm256_loadu_pd(((double*)x) + 44);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                      x_10 = _mm256_add_pd(x_10, q_2);
                      x_11 = _mm256_add_pd(x_11, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

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
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += 48){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);
                    x_8 = _mm256_loadu_pd(((double*)x) + 32);
                    x_9 = _mm256_loadu_pd(((double*)x) + 36);
                    x_10 = _mm256_loadu_pd(((double*)x) + 40);
                    x_11 = _mm256_loadu_pd(((double*)x) + 44);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                      x_10 = _mm256_add_pd(x_10, q_2);
                      x_11 = _mm256_add_pd(x_11, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);
                    x_4 = _mm256_loadu_pd(((double*)x) + 16);
                    x_5 = _mm256_loadu_pd(((double*)x) + 20);
                    x_6 = _mm256_loadu_pd(((double*)x) + 24);
                    x_7 = _mm256_loadu_pd(((double*)x) + 28);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += 32;
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);
                    x_2 = _mm256_loadu_pd(((double*)x) + 8);
                    x_3 = _mm256_loadu_pd(((double*)x) + 12);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += 16;
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));
                    x_1 = _mm256_loadu_pd(((double*)x) + 4);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += 8;
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_loadu_pd(((double*)x));

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += 4;
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += ((N_block - i) * 2);
                  }
                }
              }else{
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);
                    x_8 = _mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]);
                    x_9 = _mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]);
                    x_10 = _mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]);
                    x_11 = _mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_8, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_9, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_10, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_11, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_8 = _mm256_add_pd(_mm256_add_pd(x_8, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_9 = _mm256_add_pd(_mm256_add_pd(x_9, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_10 = _mm256_add_pd(_mm256_add_pd(x_10, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_11 = _mm256_add_pd(_mm256_add_pd(x_11, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                      x_10 = _mm256_add_pd(x_10, q_2);
                      x_11 = _mm256_add_pd(x_11, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_4, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_5, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_6, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_7, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_4 = _mm256_add_pd(_mm256_add_pd(x_4, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_5 = _mm256_add_pd(_mm256_add_pd(x_5, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_6 = _mm256_add_pd(_mm256_add_pd(x_6, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_7 = _mm256_add_pd(_mm256_add_pd(x_7, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(x_1, compression_0), blp_mask_tmp));
                    q_2 = _mm256_add_pd(s_2, _mm256_or_pd(_mm256_mul_pd(x_2, compression_0), blp_mask_tmp));
                    q_3 = _mm256_add_pd(s_3, _mm256_or_pd(_mm256_mul_pd(x_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    q_1 = _mm256_sub_pd(s_1, q_1);
                    q_2 = _mm256_sub_pd(s_2, q_2);
                    q_3 = _mm256_sub_pd(s_3, q_3);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

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
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_sub_pd(s_0, q_0);
                    x_0 = _mm256_add_pd(_mm256_add_pd(x_0, _mm256_mul_pd(q_0, expansion_0)), _mm256_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm256_add_pd(_mm256_add_pd(x_1, _mm256_mul_pd(q_1, expansion_0)), _mm256_mul_pd(q_1, expansion_mask_0));
                    x_2 = _mm256_add_pd(_mm256_add_pd(x_2, _mm256_mul_pd(q_2, expansion_0)), _mm256_mul_pd(q_2, expansion_mask_0));
                    x_3 = _mm256_add_pd(_mm256_add_pd(x_3, _mm256_mul_pd(q_3, expansion_0)), _mm256_mul_pd(q_3, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }else{
                  for(i = 0; i + 24 <= N_block; i += 24, x += (incX * 48)){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);
                    x_8 = _mm256_set_pd(((double*)x)[((incX * 34) + 1)], ((double*)x)[(incX * 34)], ((double*)x)[((incX * 32) + 1)], ((double*)x)[(incX * 32)]);
                    x_9 = _mm256_set_pd(((double*)x)[((incX * 38) + 1)], ((double*)x)[(incX * 38)], ((double*)x)[((incX * 36) + 1)], ((double*)x)[(incX * 36)]);
                    x_10 = _mm256_set_pd(((double*)x)[((incX * 42) + 1)], ((double*)x)[(incX * 42)], ((double*)x)[((incX * 40) + 1)], ((double*)x)[(incX * 40)]);
                    x_11 = _mm256_set_pd(((double*)x)[((incX * 46) + 1)], ((double*)x)[(incX * 46)], ((double*)x)[((incX * 44) + 1)], ((double*)x)[(incX * 44)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_8, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_9, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_10, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_11, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_8 = _mm256_add_pd(x_8, q_0);
                      x_9 = _mm256_add_pd(x_9, q_1);
                      x_10 = _mm256_add_pd(x_10, q_2);
                      x_11 = _mm256_add_pd(x_11, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_8, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_9, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_10, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_11, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                    x_4 = _mm256_set_pd(((double*)x)[((incX * 18) + 1)], ((double*)x)[(incX * 18)], ((double*)x)[((incX * 16) + 1)], ((double*)x)[(incX * 16)]);
                    x_5 = _mm256_set_pd(((double*)x)[((incX * 22) + 1)], ((double*)x)[(incX * 22)], ((double*)x)[((incX * 20) + 1)], ((double*)x)[(incX * 20)]);
                    x_6 = _mm256_set_pd(((double*)x)[((incX * 26) + 1)], ((double*)x)[(incX * 26)], ((double*)x)[((incX * 24) + 1)], ((double*)x)[(incX * 24)]);
                    x_7 = _mm256_set_pd(((double*)x)[((incX * 30) + 1)], ((double*)x)[(incX * 30)], ((double*)x)[((incX * 28) + 1)], ((double*)x)[(incX * 28)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_4, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_5, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_6, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_7, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_4 = _mm256_add_pd(x_4, q_0);
                      x_5 = _mm256_add_pd(x_5, q_1);
                      x_6 = _mm256_add_pd(x_6, q_2);
                      x_7 = _mm256_add_pd(x_7, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_4, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_5, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_6, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_7, blp_mask_tmp));
                    i += 16, x += (incX * 32);
                  }
                  if(i + 8 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                    x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                    x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      q_2 = _mm256_add_pd(s_2, _mm256_or_pd(x_2, blp_mask_tmp));
                      q_3 = _mm256_add_pd(s_3, _mm256_or_pd(x_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      q_2 = _mm256_sub_pd(s_2, q_2);
                      q_3 = _mm256_sub_pd(s_3, q_3);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                      x_2 = _mm256_add_pd(x_2, q_2);
                      x_3 = _mm256_add_pd(x_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm256_add_pd(s_buffer[((j * 4) + 2)], _mm256_or_pd(x_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm256_add_pd(s_buffer[((j * 4) + 3)], _mm256_or_pd(x_3, blp_mask_tmp));
                    i += 8, x += (incX * 16);
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                    x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      q_1 = _mm256_sub_pd(s_1, q_1);
                      x_0 = _mm256_add_pd(x_0, q_0);
                      x_1 = _mm256_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm256_add_pd(s_buffer[((j * 4) + 1)], _mm256_or_pd(x_1, blp_mask_tmp));
                    i += 4, x += (incX * 8);
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    i += 2, x += (incX * 4);
                  }
                  if(i < N_block){
                    x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm256_sub_pd(s_0, q_0);
                      x_0 = _mm256_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_or_pd(x_0, blp_mask_tmp));
                    x += (incX * (N_block - i) * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = _mm256_sub_pd(s_buffer[(j * 4)], _mm256_set_pd(((double*)priY)[((incpriY * j * 2) + 1)], ((double*)priY)[(incpriY * j * 2)], 0, 0));
                cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priY)) + (incpriY * j * 2)));
                s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 1)], cons_tmp));
                s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 2)], cons_tmp));
                s_buffer[(j * 4)] = _mm256_add_pd(s_buffer[(j * 4)], _mm256_sub_pd(s_buffer[((j * 4) + 3)], cons_tmp));
                _mm256_store_pd(cons_buffer_tmp, s_buffer[(j * 4)]);
                ((double*)priY)[(incpriY * j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
                ((double*)priY)[((incpriY * j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
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
              __m128d x_0, x_1, x_2, x_3;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priY));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 2));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += 8){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_3, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    q_1 = _mm_sub_pd(q_1, s_0_1);
                    x_2 = _mm_add_pd(x_2, q_0);
                    x_3 = _mm_add_pd(x_3, q_1);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_2, blp_mask_tmp));
                    s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_3, blp_mask_tmp));
                  }
                  if(i + 2 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_0, blp_mask_tmp));
                    q_0 = _mm_sub_pd(q_0, s_0_0);
                    x_0 = _mm_add_pd(x_0, q_0);
                    s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0];
              ((double*)priY)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priY));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 2));
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 4));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 6 <= N_block; i += 6, x += 12){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);
                    x_4 = _mm_loadu_pd(((double*)x) + 8);
                    x_5 = _mm_loadu_pd(((double*)x) + 10);

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                  for(i = 0; i + 6 <= N_block; i += 6, x += 12){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);
                    x_4 = _mm_loadu_pd(((double*)x) + 8);
                    x_5 = _mm_loadu_pd(((double*)x) + 10);

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                    x_4 = _mm_loadu_pd(((double*)x) + (incX * 8));
                    x_5 = _mm_loadu_pd(((double*)x) + (incX * 10));

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                  for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                    x_4 = _mm_loadu_pd(((double*)x) + (incX * 8));
                    x_5 = _mm_loadu_pd(((double*)x) + (incX * 10));

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

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

              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0];
              ((double*)priY)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 4));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priY)[(incpriY * 4)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 4) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;
              __m128d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priY));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 2));
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 4));
              s_3_0 = s_3_1 = _mm_loadu_pd(((double*)priY) + (incpriY * 6));

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 6 <= N_block; i += 6, x += 12){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);
                    x_4 = _mm_loadu_pd(((double*)x) + 8);
                    x_5 = _mm_loadu_pd(((double*)x) + 10);

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                  for(i = 0; i + 6 <= N_block; i += 6, x += 12){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);
                    x_4 = _mm_loadu_pd(((double*)x) + 8);
                    x_5 = _mm_loadu_pd(((double*)x) + 10);

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);
                    x_2 = _mm_loadu_pd(((double*)x) + 4);
                    x_3 = _mm_loadu_pd(((double*)x) + 6);

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                    x_4 = _mm_loadu_pd(((double*)x) + (incX * 8));
                    x_5 = _mm_loadu_pd(((double*)x) + (incX * 10));

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

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
                  for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                    x_4 = _mm_loadu_pd(((double*)x) + (incX * 8));
                    x_5 = _mm_loadu_pd(((double*)x) + (incX * 10));

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
                  }
                  if(i + 4 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                    x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                    x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));

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
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    x_0 = _mm_loadu_pd(((double*)x));

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

              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priY)[0] = cons_buffer_tmp[0];
              ((double*)priY)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priY)[(incpriY * 2)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 2) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 4));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priY)[(incpriY * 4)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 4) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * 6));
              s_3_0 = _mm_add_pd(s_3_0, _mm_sub_pd(s_3_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)priY)[(incpriY * 6)] = cons_buffer_tmp[0];
              ((double*)priY)[((incpriY * 6) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128d x_0, x_1;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0, s_1;
              __m128d s_buffer[(binned_DBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm_loadu_pd(((double*)priY) + (incpriY * j * 2));
              }

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 2 <= N_block; i += 2, x += 4){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

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
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }else{
                  for(i = 0; i + 2 <= N_block; i += 2, x += 4){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + 2);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += 2;
                  }
                }
              }else{
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                  for(i = 0; i + 2 <= N_block; i += 2, x += (incX * 4)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

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
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_sub_pd(s_0, q_0);
                    x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                    x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }else{
                  for(i = 0; i + 2 <= N_block; i += 2, x += (incX * 4)){
                    x_0 = _mm_loadu_pd(((double*)x));
                    x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      s_1 = s_buffer[((j * 2) + 1)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      s_buffer[((j * 2) + 1)] = q_1;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      x_0 = _mm_add_pd(x_0, q_0);
                      x_1 = _mm_add_pd(x_1, q_1);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                  }
                  if(i + 1 <= N_block){
                    x_0 = _mm_loadu_pd(((double*)x));

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 2)];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[(j * 2)] = q_0;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      x_0 = _mm_add_pd(x_0, q_0);
                    }
                    s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(x_0, blp_mask_tmp));
                    i += 1, x += (incX * 2);
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                cons_tmp = _mm_loadu_pd(((double*)((double*)priY)) + (incpriY * j * 2));
                s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_sub_pd(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm_store_pd(cons_buffer_tmp, s_buffer[(j * 2)]);
                ((double*)priY)[(incpriY * j * 2)] = cons_buffer_tmp[0];
                ((double*)priY)[((incpriY * j * 2) + 1)] = cons_buffer_tmp[1];
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

              s_0_0 = ((double*)priY)[0];
              s_0_1 = ((double*)priY)[1];
              s_1_0 = ((double*)priY)[(incpriY * 2)];
              s_1_1 = ((double*)priY)[((incpriY * 2) + 1)];
              s_2_0 = ((double*)priY)[(incpriY * 4)];
              s_2_1 = ((double*)priY)[((incpriY * 4) + 1)];

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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

              ((double*)priY)[0] = s_0_0;
              ((double*)priY)[1] = s_0_1;
              ((double*)priY)[(incpriY * 2)] = s_1_0;
              ((double*)priY)[((incpriY * 2) + 1)] = s_1_1;
              ((double*)priY)[(incpriY * 4)] = s_2_0;
              ((double*)priY)[((incpriY * 4) + 1)] = s_2_1;

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
                s_buffer[(j * 2)] = ((double*)priY)[(incpriY * j * 2)];
                s_buffer[((j * 2) + 1)] = ((double*)priY)[((incpriY * j * 2) + 1)];
              }

              if(incX == 1){
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                if(binned_dmindex0(priY) || binned_dmindex0(priY + 1)){
                  if(binned_dmindex0(priY)){
                    if(binned_dmindex0(priY + 1)){
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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                    x_0 = ((double*)x)[0];
                    x_1 = ((double*)x)[1];

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
                ((double*)priY)[(incpriY * j * 2)] = s_buffer[(j * 2)];
                ((double*)priY)[((incpriY * j * 2) + 1)] = s_buffer[((j * 2) + 1)];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    if (isinf(amax[0])){
      priY[0] = amax[0];
    }
    if (isinf(amax[1])){
      priY[1] = amax[1];
    }

    deposits += N_block;
  }

  binned_zmrenorm(fold, priY, incpriY, carY, inccarY);
}