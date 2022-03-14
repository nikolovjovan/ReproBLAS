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

cog.out(generate.generate(blockSize.BlockSize("smssum", "N_block_MAX", 32, terminal.get_siendurance(), terminal.get_siendurance(), ["bench_rssum_fold_{}".format(terminal.get_sidefaultfold())]), cog.inFile, args, params, mode))

]]]*/
#if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
  #include <immintrin.h>

#elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
  #include <emmintrin.h>

#else


#endif

#define N_block_MAX 512
//[[[end]]]

/**
 * @internal
 * @brief Add to manually specified binned single precision Y the sum of single precision vector X
 *
 * Add to Y the binned sum of X.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X single precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param priY Y's primary vector
 * @param incpriY stride within Y's primary vector (use every incpriY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_smssum(const int fold, const int N, const float *X, const int incX, float *priY, const int incpriY, float *carY, const int inccarY){
  float amax;
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    amax = binnedBLAS_samax(N_block, X, incX);

    if (isinf(amax) || isinf(priY[0])){
      for (j = 0; j < N_block; j++){
        priY[0] += X[j * incX];
      }
    }
    if (isnan(priY[0])){
      return;
    } else if (isinf(priY[0])){
      X += N_block * incX;
      continue;
    }

    if (deposits + N_block > binned_SBENDURANCE) {
      binned_smrenorm(fold, priY, incpriY, carY, inccarY);
      deposits = 0;
    }

    binned_smsupdate(fold, amax, priY, incpriY, carY, inccarY);

    /*[[[cog
    cog.out(generate.generate(depositSum.DepositSum(dataTypes.Float, "fold", "N_block", "X", "incX", "priY", "incpriY"), cog.inFile, args, params, mode))
    ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
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
              __m256 X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm256_broadcast_ss(priY);
              s_1_0 = s_1_1 = _mm256_broadcast_ss(priY + incpriY);

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 80 <= N_block; i += 80, X += 80){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);
                    X_8 = _mm256_loadu_ps(X + 64);
                    X_9 = _mm256_loadu_ps(X + 72);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_8 = _mm256_add_ps(_mm256_add_ps(X_8, q_0), q_0);
                    X_9 = _mm256_add_ps(_mm256_add_ps(X_9, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_9, blp_mask_tmp));
                  }
                  if(i + 64 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    i += 64, X += 64;
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 80 <= N_block; i += 80, X += 80){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);
                    X_8 = _mm256_loadu_ps(X + 64);
                    X_9 = _mm256_loadu_ps(X + 72);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    X_7 = _mm256_add_ps(X_7, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_9, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_8 = _mm256_add_ps(X_8, q_0);
                    X_9 = _mm256_add_ps(X_9, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_9, blp_mask_tmp));
                  }
                  if(i + 64 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    X_7 = _mm256_add_ps(X_7, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    i += 64, X += 64;
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 80 <= N_block; i += 80, X += (incX * 80)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);
                    X_8 = _mm256_set_ps(X[(incX * 71)], X[(incX * 70)], X[(incX * 69)], X[(incX * 68)], X[(incX * 67)], X[(incX * 66)], X[(incX * 65)], X[(incX * 64)]);
                    X_9 = _mm256_set_ps(X[(incX * 79)], X[(incX * 78)], X[(incX * 77)], X[(incX * 76)], X[(incX * 75)], X[(incX * 74)], X[(incX * 73)], X[(incX * 72)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_8, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_9, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_8 = _mm256_add_ps(_mm256_add_ps(X_8, q_0), q_0);
                    X_9 = _mm256_add_ps(_mm256_add_ps(X_9, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_9, blp_mask_tmp));
                  }
                  if(i + 64 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    i += 64, X += (incX * 64);
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 80 <= N_block; i += 80, X += (incX * 80)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);
                    X_8 = _mm256_set_ps(X[(incX * 71)], X[(incX * 70)], X[(incX * 69)], X[(incX * 68)], X[(incX * 67)], X[(incX * 66)], X[(incX * 65)], X[(incX * 64)]);
                    X_9 = _mm256_set_ps(X[(incX * 79)], X[(incX * 78)], X[(incX * 77)], X[(incX * 76)], X[(incX * 75)], X[(incX * 74)], X[(incX * 73)], X[(incX * 72)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    X_7 = _mm256_add_ps(X_7, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_9, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_8 = _mm256_add_ps(X_8, q_0);
                    X_9 = _mm256_add_ps(X_9, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_8, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_9, blp_mask_tmp));
                  }
                  if(i + 64 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    X_7 = _mm256_add_ps(X_7, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_7, blp_mask_tmp));
                    i += 64, X += (incX * 64);
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], 0));
              cons_tmp = _mm256_broadcast_ss(priY);
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], 0));
              cons_tmp = _mm256_broadcast_ss(priY + incpriY);
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256 X_0, X_1, X_2, X_3, X_4, X_5;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;
              __m256 s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm256_broadcast_ss(priY);
              s_1_0 = s_1_1 = _mm256_broadcast_ss(priY + incpriY);
              s_2_0 = s_2_1 = _mm256_broadcast_ss(priY + (incpriY * 2));

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 48 <= N_block; i += 48, X += 48){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_5, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 48 <= N_block; i += 48, X += 48){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_5, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 48 <= N_block; i += 48, X += (incX * 48)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_5, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm256_mul_ps(_mm256_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 48 <= N_block; i += 48, X += (incX * 48)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    X_5 = _mm256_add_ps(X_5, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_5, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    X_3 = _mm256_add_ps(X_3, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    q_1 = _mm256_sub_ps(q_1, s_0_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    q_1 = _mm256_sub_ps(q_1, s_1_1);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    X_1 = _mm256_add_ps(X_1, q_1);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], 0));
              cons_tmp = _mm256_broadcast_ss(priY);
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], 0));
              cons_tmp = _mm256_broadcast_ss(priY + incpriY);
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], 0));
              cons_tmp = _mm256_broadcast_ss(priY + (incpriY * 2));
              s_2_0 = _mm256_add_ps(s_2_0, _mm256_sub_ps(s_2_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              priY[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256 X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 q_0;
              __m256 s_0_0;
              __m256 s_1_0;
              __m256 s_2_0;
              __m256 s_3_0;

              s_0_0 = _mm256_broadcast_ss(priY);
              s_1_0 = _mm256_broadcast_ss(priY + incpriY);
              s_2_0 = _mm256_broadcast_ss(priY + (incpriY * 2));
              s_3_0 = _mm256_broadcast_ss(priY + (incpriY * 3));

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 64 <= N_block; i += 64, X += 64){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 64 <= N_block; i += 64, X += 64){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);
                    X_4 = _mm256_loadu_ps(X + 32);
                    X_5 = _mm256_loadu_ps(X + 40);
                    X_6 = _mm256_loadu_ps(X + 48);
                    X_7 = _mm256_loadu_ps(X + 56);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);
                    X_2 = _mm256_loadu_ps(X + 16);
                    X_3 = _mm256_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += 32;
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 64 <= N_block; i += 64, X += (incX * 64)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_4, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_4 = _mm256_add_ps(_mm256_add_ps(X_4, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_5 = _mm256_add_ps(_mm256_add_ps(X_5, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_6, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_6 = _mm256_add_ps(_mm256_add_ps(X_6, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_7 = _mm256_add_ps(_mm256_add_ps(X_7, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm256_add_ps(_mm256_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm256_add_ps(_mm256_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 64 <= N_block; i += 64, X += (incX * 64)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_4 = _mm256_set_ps(X[(incX * 39)], X[(incX * 38)], X[(incX * 37)], X[(incX * 36)], X[(incX * 35)], X[(incX * 34)], X[(incX * 33)], X[(incX * 32)]);
                    X_5 = _mm256_set_ps(X[(incX * 47)], X[(incX * 46)], X[(incX * 45)], X[(incX * 44)], X[(incX * 43)], X[(incX * 42)], X[(incX * 41)], X[(incX * 40)]);
                    X_6 = _mm256_set_ps(X[(incX * 55)], X[(incX * 54)], X[(incX * 53)], X[(incX * 52)], X[(incX * 51)], X[(incX * 50)], X[(incX * 49)], X[(incX * 48)]);
                    X_7 = _mm256_set_ps(X[(incX * 63)], X[(incX * 62)], X[(incX * 61)], X[(incX * 60)], X[(incX * 59)], X[(incX * 58)], X[(incX * 57)], X[(incX * 56)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_4 = _mm256_add_ps(X_4, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_5 = _mm256_add_ps(X_5, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_6 = _mm256_add_ps(X_6, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_6, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_7 = _mm256_add_ps(X_7, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 32 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_2 = _mm256_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)], X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_3 = _mm256_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)], X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_2 = _mm256_add_ps(X_2, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_3 = _mm256_add_ps(X_3, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_3, blp_mask_tmp));
                    i += 32, X += (incX * 32);
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_1 = _mm256_add_ps(X_1, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_1, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_0_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_1_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm256_sub_ps(q_0, s_2_0);
                    X_0 = _mm256_add_ps(X_0, q_0);
                    s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], priY[0], 0));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], priY[incpriY], 0));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], 0));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              priY[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              s_3_0 = _mm256_sub_ps(s_3_0, _mm256_set_ps(priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], 0));
              _mm256_store_ps(cons_buffer_tmp, s_3_0);
              priY[(incpriY * 3)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256 X_0, X_1;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 q_0;
              __m256 s_0;
              __m256 s_buffer[binned_SBMAXFOLD];

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = _mm256_broadcast_ss(priY + (incpriY * j));
              }

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_1 = _mm256_add_ps(X_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_1, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm256_loadu_ps(X);
                    X_1 = _mm256_loadu_ps(X + 8);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_1 = _mm256_add_ps(X_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_1, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_loadu_ps(X);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[6]:0, (N_block - i)>5?X[5]:0, (N_block - i)>4?X[4]:0, (N_block - i)>3?X[3]:0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm256_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm256_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_1 = _mm256_add_ps(_mm256_add_ps(X_1, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_1 = _mm256_add_ps(X_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_1, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm256_mul_ps(_mm256_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm256_add_ps(_mm256_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm256_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)], X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_1, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_1 = _mm256_add_ps(X_1, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_1, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm256_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)], X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i < N_block){
                    X_0 = _mm256_set_ps(0, (N_block - i)>6?X[(incX * 6)]:0, (N_block - i)>5?X[(incX * 5)]:0, (N_block - i)>4?X[(incX * 4)]:0, (N_block - i)>3?X[(incX * 3)]:0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(X_0, blp_mask_tmp));
                      s_buffer[j] = q_0;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      X_0 = _mm256_add_ps(X_0, q_0);
                    }
                    s_buffer[j] = _mm256_add_ps(s_buffer[j], _mm256_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = _mm256_sub_ps(s_buffer[j], _mm256_set_ps(priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], 0));
                _mm256_store_ps(cons_buffer_tmp, s_buffer[j]);
                priY[(incpriY * j)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3] + cons_buffer_tmp[4] + cons_buffer_tmp[5] + cons_buffer_tmp[6] + cons_buffer_tmp[7];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
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
              __m128 X_0, X_1, X_2, X_3;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm_load1_ps(priY);
              s_1_0 = s_1_1 = _mm_load1_ps(priY + incpriY);

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(priY[0], priY[0], priY[0], 0));
              cons_tmp = _mm_load1_ps(priY);
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], 0));
              cons_tmp = _mm_load1_ps(priY + incpriY);
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128 X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;
              __m128 s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm_load1_ps(priY);
              s_1_0 = s_1_1 = _mm_load1_ps(priY + incpriY);
              s_2_0 = s_2_1 = _mm_load1_ps(priY + (incpriY * 2));

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 32 <= N_block; i += 32, X += 32){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);
                    X_4 = _mm_loadu_ps(X + 16);
                    X_5 = _mm_loadu_ps(X + 20);
                    X_6 = _mm_loadu_ps(X + 24);
                    X_7 = _mm_loadu_ps(X + 28);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm_add_ps(_mm_add_ps(X_4, q_0), q_0);
                    X_5 = _mm_add_ps(_mm_add_ps(X_5, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm_add_ps(_mm_add_ps(X_6, q_0), q_0);
                    X_7 = _mm_add_ps(_mm_add_ps(X_7, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 32 <= N_block; i += 32, X += 32){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);
                    X_4 = _mm_loadu_ps(X + 16);
                    X_5 = _mm_loadu_ps(X + 20);
                    X_6 = _mm_loadu_ps(X + 24);
                    X_7 = _mm_loadu_ps(X + 28);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                    X_4 = _mm_set_ps(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_5 = _mm_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                    X_6 = _mm_set_ps(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_7 = _mm_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_4, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_4 = _mm_add_ps(_mm_add_ps(X_4, q_0), q_0);
                    X_5 = _mm_add_ps(_mm_add_ps(X_5, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_6, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_7, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_6 = _mm_add_ps(_mm_add_ps(X_6, q_0), q_0);
                    X_7 = _mm_add_ps(_mm_add_ps(X_7, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(q_1, s_0_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                    X_4 = _mm_set_ps(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_5 = _mm_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                    X_6 = _mm_set_ps(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                    X_7 = _mm_set_ps(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_4 = _mm_add_ps(X_4, q_0);
                    X_5 = _mm_add_ps(X_5, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_7, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_6 = _mm_add_ps(X_6, q_0);
                    X_7 = _mm_add_ps(X_7, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_7, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_2 = _mm_add_ps(X_2, q_0);
                    X_3 = _mm_add_ps(X_3, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    q_1 = s_0_1;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    q_1 = _mm_sub_ps(q_1, s_0_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    q_0 = s_1_0;
                    q_1 = s_1_1;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    q_1 = _mm_sub_ps(q_1, s_1_1);
                    X_0 = _mm_add_ps(X_0, q_0);
                    X_1 = _mm_add_ps(X_1, q_1);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(priY[0], priY[0], priY[0], 0));
              cons_tmp = _mm_load1_ps(priY);
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], 0));
              cons_tmp = _mm_load1_ps(priY + incpriY);
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], 0));
              cons_tmp = _mm_load1_ps(priY + (incpriY * 2));
              s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              priY[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128 X_0, X_1, X_2, X_3, X_4, X_5, X_6;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 q_0;
              __m128 s_0_0;
              __m128 s_1_0;
              __m128 s_2_0;
              __m128 s_3_0;

              s_0_0 = _mm_load1_ps(priY);
              s_1_0 = _mm_load1_ps(priY + incpriY);
              s_2_0 = _mm_load1_ps(priY + (incpriY * 2));
              s_3_0 = _mm_load1_ps(priY + (incpriY * 3));

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 28 <= N_block; i += 28, X += 28){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);
                    X_4 = _mm_loadu_ps(X + 16);
                    X_5 = _mm_loadu_ps(X + 20);
                    X_6 = _mm_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_4, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_4 = _mm_add_ps(_mm_add_ps(X_4, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_5 = _mm_add_ps(_mm_add_ps(X_5, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_6, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_6 = _mm_add_ps(_mm_add_ps(X_6, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_6, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, X += 28){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);
                    X_4 = _mm_loadu_ps(X + 16);
                    X_5 = _mm_loadu_ps(X + 20);
                    X_6 = _mm_loadu_ps(X + 24);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_6, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += 16;
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 28 <= N_block; i += 28, X += (incX * 28)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                    X_4 = _mm_set_ps(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_5 = _mm_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                    X_6 = _mm_set_ps(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_4, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_4 = _mm_add_ps(_mm_add_ps(X_4, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_5, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_5 = _mm_add_ps(_mm_add_ps(X_5, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_6, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_6 = _mm_add_ps(_mm_add_ps(X_6, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_6, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_0 = _mm_mul_ps(_mm_sub_ps(q_0, s_0_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 28 <= N_block; i += 28, X += (incX * 28)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                    X_4 = _mm_set_ps(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                    X_5 = _mm_set_ps(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                    X_6 = _mm_set_ps(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_4 = _mm_add_ps(X_4, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_4, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_5 = _mm_add_ps(X_5, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_5, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_6, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_6 = _mm_add_ps(X_6, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_6, blp_mask_tmp));
                  }
                  if(i + 16 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_2 = _mm_add_ps(X_2, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_2, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_3, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_3 = _mm_add_ps(X_3, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_3, blp_mask_tmp));
                    i += 16, X += (incX * 16);
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_1, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_1 = _mm_add_ps(X_1, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    q_0 = s_0_0;
                    s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_0_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_1_0;
                    s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_1_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    q_0 = s_2_0;
                    s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(X_0, blp_mask_tmp));
                    q_0 = _mm_sub_ps(q_0, s_2_0);
                    X_0 = _mm_add_ps(X_0, q_0);
                    s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(priY[0], priY[0], priY[0], 0));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              priY[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(priY[incpriY], priY[incpriY], priY[incpriY], 0));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              priY[incpriY] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(priY[(incpriY * 2)], priY[(incpriY * 2)], priY[(incpriY * 2)], 0));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              priY[(incpriY * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_3_0 = _mm_sub_ps(s_3_0, _mm_set_ps(priY[(incpriY * 3)], priY[(incpriY * 3)], priY[(incpriY * 3)], 0));
              _mm_store_ps(cons_buffer_tmp, s_3_0);
              priY[(incpriY * 3)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128 X_0, X_1, X_2, X_3;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 q_0, q_1, q_2, q_3;
              __m128 s_0, s_1, s_2, s_3;
              __m128 s_buffer[(binned_SBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = _mm_load1_ps(priY + (incpriY * j));
              }

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(s_1, q_1), expansion_0);
                    q_2 = _mm_mul_ps(_mm_sub_ps(s_2, q_2), expansion_0);
                    q_3 = _mm_mul_ps(_mm_sub_ps(s_3, q_3), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_2), q_2);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_3), q_3);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(X_2, blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(X_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                      X_2 = _mm_add_ps(X_2, q_2);
                      X_3 = _mm_add_ps(X_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(X_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(s_1, q_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += 16){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);
                    X_2 = _mm_loadu_ps(X + 8);
                    X_3 = _mm_loadu_ps(X + 12);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(X_2, blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(X_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                      X_2 = _mm_add_ps(X_2, q_2);
                      X_3 = _mm_add_ps(X_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(X_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_loadu_ps(X);
                    X_1 = _mm_loadu_ps(X + 4);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += 8;
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_loadu_ps(X);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += 4;
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    X += (N_block - i);
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = _mm_set1_ps(binned_SMCOMPRESSION);
                  expansion_0 = _mm_set1_ps(binned_SMEXPANSION * 0.5);
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    s_2 = s_buffer[2];
                    s_3 = s_buffer[3];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(X_2, compression_0), blp_mask_tmp));
                    q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(X_3, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    s_buffer[2] = q_2;
                    s_buffer[3] = q_3;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(s_1, q_1), expansion_0);
                    q_2 = _mm_mul_ps(_mm_sub_ps(s_2, q_2), expansion_0);
                    q_3 = _mm_mul_ps(_mm_sub_ps(s_3, q_3), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    X_2 = _mm_add_ps(_mm_add_ps(X_2, q_2), q_2);
                    X_3 = _mm_add_ps(_mm_add_ps(X_3, q_3), q_3);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(X_2, blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(X_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                      X_2 = _mm_add_ps(X_2, q_2);
                      X_3 = _mm_add_ps(X_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(X_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    s_0 = s_buffer[0];
                    s_1 = s_buffer[1];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(X_1, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    s_buffer[1] = q_1;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    q_1 = _mm_mul_ps(_mm_sub_ps(s_1, q_1), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    X_1 = _mm_add_ps(_mm_add_ps(X_1, q_1), q_1);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    s_0 = s_buffer[0];
                    q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(X_0, compression_0), blp_mask_tmp));
                    s_buffer[0] = q_0;
                    q_0 = _mm_mul_ps(_mm_sub_ps(s_0, q_0), expansion_0);
                    X_0 = _mm_add_ps(_mm_add_ps(X_0, q_0), q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }else{
                  for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16)){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                    X_2 = _mm_set_ps(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                    X_3 = _mm_set_ps(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      s_2 = s_buffer[((j * 4) + 2)];
                      s_3 = s_buffer[((j * 4) + 3)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(X_2, blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(X_3, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      s_buffer[((j * 4) + 2)] = q_2;
                      s_buffer[((j * 4) + 3)] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                      X_2 = _mm_add_ps(X_2, q_2);
                      X_3 = _mm_add_ps(X_3, q_3);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(X_2, blp_mask_tmp));
                    s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(X_3, blp_mask_tmp));
                  }
                  if(i + 8 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                    X_1 = _mm_set_ps(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      s_1 = s_buffer[((j * 4) + 1)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(X_1, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      s_buffer[((j * 4) + 1)] = q_1;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      X_0 = _mm_add_ps(X_0, q_0);
                      X_1 = _mm_add_ps(X_1, q_1);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(X_1, blp_mask_tmp));
                    i += 8, X += (incX * 8);
                  }
                  if(i + 4 <= N_block){
                    X_0 = _mm_set_ps(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    i += 4, X += (incX * 4);
                  }
                  if(i < N_block){
                    X_0 = _mm_set_ps(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[(j * 4)];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(X_0, blp_mask_tmp));
                      s_buffer[(j * 4)] = q_0;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      X_0 = _mm_add_ps(X_0, q_0);
                    }
                    s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(X_0, blp_mask_tmp));
                    X += (incX * (N_block - i));
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = _mm_sub_ps(s_buffer[(j * 4)], _mm_set_ps(priY[(incpriY * j)], priY[(incpriY * j)], priY[(incpriY * j)], 0));
                cons_tmp = _mm_load1_ps(priY + (incpriY * j));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 1)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 2)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 3)], cons_tmp));
                _mm_store_ps(cons_buffer_tmp, s_buffer[(j * 4)]);
                priY[(incpriY * j)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
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
              float X_0;
              float compression_0;
              float expansion_0;
              float q_0;
              float s_0_0;
              float s_1_0;
              float s_2_0;

              s_0_0 = priY[0];
              s_1_0 = priY[incpriY];
              s_2_0 = priY[(incpriY * 2)];

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = binned_SMCOMPRESSION;
                  expansion_0 = binned_SMEXPANSION * 0.5;
                  for(i = 0; i + 1 <= N_block; i += 1, X += 1){
                    X_0 = X[0];

                    q_0 = s_0_0;
                    blp_tmp.f = (X_0 * compression_0);
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    q_0 = ((q_0 - s_0_0) * expansion_0);
                    X_0 = ((X_0 + q_0) + q_0);
                    q_0 = s_1_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    X_0 = (X_0 + q_0);
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, X += 1){
                    X_0 = X[0];

                    q_0 = s_0_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    X_0 = (X_0 + q_0);
                    q_0 = s_1_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    X_0 = (X_0 + q_0);
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = binned_SMCOMPRESSION;
                  expansion_0 = binned_SMEXPANSION * 0.5;
                  for(i = 0; i + 1 <= N_block; i += 1, X += incX){
                    X_0 = X[0];

                    q_0 = s_0_0;
                    blp_tmp.f = (X_0 * compression_0);
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    q_0 = ((q_0 - s_0_0) * expansion_0);
                    X_0 = ((X_0 + q_0) + q_0);
                    q_0 = s_1_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    X_0 = (X_0 + q_0);
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, X += incX){
                    X_0 = X[0];

                    q_0 = s_0_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_0_0 = s_0_0 + blp_tmp.f;
                    q_0 = (q_0 - s_0_0);
                    X_0 = (X_0 + q_0);
                    q_0 = s_1_0;
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_1_0 = s_1_0 + blp_tmp.f;
                    q_0 = (q_0 - s_1_0);
                    X_0 = (X_0 + q_0);
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_2_0 = s_2_0 + blp_tmp.f;
                  }
                }
              }

              priY[0] = s_0_0;
              priY[incpriY] = s_1_0;
              priY[(incpriY * 2)] = s_2_0;

            }
            break;
          default:
            {
              int i, j;
              float X_0;
              float compression_0;
              float expansion_0;
              float q_0;
              float s_0;
              float s_buffer[binned_SBMAXFOLD];

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = priY[(incpriY * j)];
              }

              if(incX == 1){
                if(binned_smindex0(priY)){
                  compression_0 = binned_SMCOMPRESSION;
                  expansion_0 = binned_SMEXPANSION * 0.5;
                  for(i = 0; i + 1 <= N_block; i += 1, X += 1){
                    X_0 = X[0];

                    s_0 = s_buffer[0];
                    blp_tmp.f = (X_0 * compression_0);
                    blp_tmp.i |= 1;
                    q_0 = s_0 + blp_tmp.f;
                    s_buffer[0] = q_0;
                    q_0 = ((s_0 - q_0) * expansion_0);
                    X_0 = ((X_0 + q_0) + q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      blp_tmp.f = X_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      s_buffer[j] = q_0;
                      q_0 = (s_0 - q_0);
                      X_0 = (X_0 + q_0);
                    }
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_buffer[j] = s_buffer[j] + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, X += 1){
                    X_0 = X[0];

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      blp_tmp.f = X_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      s_buffer[j] = q_0;
                      q_0 = (s_0 - q_0);
                      X_0 = (X_0 + q_0);
                    }
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_buffer[j] = s_buffer[j] + blp_tmp.f;
                  }
                }
              }else{
                if(binned_smindex0(priY)){
                  compression_0 = binned_SMCOMPRESSION;
                  expansion_0 = binned_SMEXPANSION * 0.5;
                  for(i = 0; i + 1 <= N_block; i += 1, X += incX){
                    X_0 = X[0];

                    s_0 = s_buffer[0];
                    blp_tmp.f = (X_0 * compression_0);
                    blp_tmp.i |= 1;
                    q_0 = s_0 + blp_tmp.f;
                    s_buffer[0] = q_0;
                    q_0 = ((s_0 - q_0) * expansion_0);
                    X_0 = ((X_0 + q_0) + q_0);
                    for(j = 1; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      blp_tmp.f = X_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      s_buffer[j] = q_0;
                      q_0 = (s_0 - q_0);
                      X_0 = (X_0 + q_0);
                    }
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_buffer[j] = s_buffer[j] + blp_tmp.f;
                  }
                }else{
                  for(i = 0; i + 1 <= N_block; i += 1, X += incX){
                    X_0 = X[0];

                    for(j = 0; j < fold - 1; j++){
                      s_0 = s_buffer[j];
                      blp_tmp.f = X_0;
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      s_buffer[j] = q_0;
                      q_0 = (s_0 - q_0);
                      X_0 = (X_0 + q_0);
                    }
                    blp_tmp.f = X_0;
                    blp_tmp.i |= 1;
                    s_buffer[j] = s_buffer[j] + blp_tmp.f;
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                priY[(incpriY * j)] = s_buffer[j];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    deposits += N_block;
  }

  binned_smrenorm(fold, priY, incpriY, carY, inccarY);
}