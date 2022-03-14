#include <stdlib.h>
#include <math.h>

#include "../config.h"
#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
import depositDot
import vectorizations
from src.common import blockSize
from scripts import terminal

code_block = generate.CodeBlock()
vectorizations.conditionally_include_vectorizations(code_block)
cog.out(str(code_block))

cog.outl()

cog.out(generate.generate(blockSize.BlockSize("dmddot", "N_block_MAX", 32, terminal.get_diendurance(), terminal.get_diendurance(), ["bench_rddot_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
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
 * @brief Add to manually specified binned double precision Z the dot product of double precision vectors X and Y
 *
 * Add to Z the binned sum of the pairwise products of X and Y.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X double precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param Y double precision vector
 * @param incY Y vector stride (use every incY'th element)
 * @param priZ Z's primary vector
 * @param incpriZ stride within Z's primary vector (use every incpriZ'th element)
 * @param carZ Z's carry vector
 * @param inccarZ stride within Z's carry vector (use every inccarZ'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_dmddot(const int fold, const int N, const double *X, const int incX, const double *Y, const int incY, double *priZ, const int incpriZ, double *carZ, const int inccarZ){
  double amaxm;
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    amaxm = binnedBLAS_damaxm(N_block, X, incX, Y, incY);

    if (isinf(amaxm) || isinf(priZ[0])){
      for (j = 0; j < N_block; j++){
        priZ[0] += X[j * incX] * Y[j * incY];
      }
    }
    if (isnan(priZ[0])){
      return;
    } else if (isinf(priZ[0])){
      X += N_block * incX;
      Y += N_block * incY;
      continue;
    }

    if (deposits + N_block > binned_DBENDURANCE) {
      binned_dmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
      deposits = 0;
    }

    binned_dmdupdate(fold, amaxm, priZ, incpriZ, carZ, inccarZ);

    /*[[[cog
    cog.out(generate.generate(depositDot.DepositDot(dataTypes.Double, "fold", "N_block", "X", "incX", "priZ", "incpriZ", "Y", "incY"), cog.inFile, args, params, mode))
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
              __m256d X_0, X_1, X_2, X_3;
              __m256d Y_0, Y_1, Y_2, Y_3;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm256_broadcast_sd(priZ);
              s_1_0 = s_1_1 = _mm256_broadcast_sd(priZ + incpriZ);

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(priZ[0], priZ[0], priZ[0], 0));
              cons_tmp = _mm256_broadcast_sd(priZ);
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(priZ[incpriZ], priZ[incpriZ], priZ[incpriZ], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + incpriZ);
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256d X_0, X_1, X_2, X_3;
              __m256d Y_0, Y_1, Y_2, Y_3;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm256_broadcast_sd(priZ);
              s_1_0 = s_1_1 = _mm256_broadcast_sd(priZ + incpriZ);
              s_2_0 = s_2_1 = _mm256_broadcast_sd(priZ + (incpriZ * 2));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(priZ[0], priZ[0], priZ[0], 0));
              cons_tmp = _mm256_broadcast_sd(priZ);
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(priZ[incpriZ], priZ[incpriZ], priZ[incpriZ], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + incpriZ);
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(priZ[(incpriZ * 2)], priZ[(incpriZ * 2)], priZ[(incpriZ * 2)], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + (incpriZ * 2));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              priZ[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256d X_0, X_1, X_2, X_3;
              __m256d Y_0, Y_1, Y_2, Y_3;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;
              __m256d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm256_broadcast_sd(priZ);
              s_1_0 = s_1_1 = _mm256_broadcast_sd(priZ + incpriZ);
              s_2_0 = s_2_1 = _mm256_broadcast_sd(priZ + (incpriZ * 2));
              s_3_0 = s_3_1 = _mm256_broadcast_sd(priZ + (incpriZ * 3));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_2 = _mm256_add_pd(X_2, q_0);
                      X_3 = _mm256_add_pd(X_3, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      q_1 = _mm256_sub_pd(q_1, s_0_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      q_1 = _mm256_sub_pd(q_1, s_1_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      q_1 = _mm256_sub_pd(q_1, s_2_1);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      X_1 = _mm256_add_pd(X_1, q_1);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_pd(s_3_1, _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_0_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_1_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm256_sub_pd(q_0, s_2_0);
                      X_0 = _mm256_add_pd(X_0, q_0);
                      s_3_0 = _mm256_add_pd(s_3_0, _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(priZ[0], priZ[0], priZ[0], 0));
              cons_tmp = _mm256_broadcast_sd(priZ);
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(priZ[incpriZ], priZ[incpriZ], priZ[incpriZ], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + incpriZ);
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(priZ[(incpriZ * 2)], priZ[(incpriZ * 2)], priZ[(incpriZ * 2)], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + (incpriZ * 2));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              priZ[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
              s_3_0 = _mm256_sub_pd(s_3_0, _mm256_set_pd(priZ[(incpriZ * 3)], priZ[(incpriZ * 3)], priZ[(incpriZ * 3)], 0));
              cons_tmp = _mm256_broadcast_sd(priZ + (incpriZ * 3));
              s_3_0 = _mm256_add_pd(s_3_0, _mm256_sub_pd(s_3_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_3_0);
              priZ[(incpriZ * 3)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256d X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7;
              __m256d Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d q_0, q_1;
              __m256d s_0, s_1;
              __m256d s_buffer[(binned_DBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm256_broadcast_sd(priZ + (incpriZ * j));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 32 <= N_block; i += 32, X += 32, Y += 32){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      X_4 = _mm256_loadu_pd(X + 16);
                      X_5 = _mm256_loadu_pd(X + 20);
                      X_6 = _mm256_loadu_pd(X + 24);
                      X_7 = _mm256_loadu_pd(X + 28);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      Y_4 = _mm256_loadu_pd(Y + 16);
                      Y_5 = _mm256_loadu_pd(Y + 20);
                      Y_6 = _mm256_loadu_pd(Y + 24);
                      Y_7 = _mm256_loadu_pd(Y + 28);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_4, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_4 = _mm256_add_pd(_mm256_add_pd(X_4, q_0), q_0);
                      X_5 = _mm256_add_pd(_mm256_add_pd(X_5, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_6, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_6 = _mm256_add_pd(_mm256_add_pd(X_6, q_0), q_0);
                      X_7 = _mm256_add_pd(_mm256_add_pd(X_7, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += 16, Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 32 <= N_block; i += 32, X += 32, Y += 32){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      X_4 = _mm256_loadu_pd(X + 16);
                      X_5 = _mm256_loadu_pd(X + 20);
                      X_6 = _mm256_loadu_pd(X + 24);
                      X_7 = _mm256_loadu_pd(X + 28);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      Y_4 = _mm256_loadu_pd(Y + 16);
                      Y_5 = _mm256_loadu_pd(Y + 20);
                      Y_6 = _mm256_loadu_pd(Y + 24);
                      Y_7 = _mm256_loadu_pd(Y + 28);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += 16, Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 32 <= N_block; i += 32, X += 32, Y += (incY * 32)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      X_4 = _mm256_loadu_pd(X + 16);
                      X_5 = _mm256_loadu_pd(X + 20);
                      X_6 = _mm256_loadu_pd(X + 24);
                      X_7 = _mm256_loadu_pd(X + 28);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      Y_4 = _mm256_set_pd(Y[(incY * 19)], Y[(incY * 18)], Y[(incY * 17)], Y[(incY * 16)]);
                      Y_5 = _mm256_set_pd(Y[(incY * 23)], Y[(incY * 22)], Y[(incY * 21)], Y[(incY * 20)]);
                      Y_6 = _mm256_set_pd(Y[(incY * 27)], Y[(incY * 26)], Y[(incY * 25)], Y[(incY * 24)]);
                      Y_7 = _mm256_set_pd(Y[(incY * 31)], Y[(incY * 30)], Y[(incY * 29)], Y[(incY * 28)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_4, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_4 = _mm256_add_pd(_mm256_add_pd(X_4, q_0), q_0);
                      X_5 = _mm256_add_pd(_mm256_add_pd(X_5, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_6, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_6 = _mm256_add_pd(_mm256_add_pd(X_6, q_0), q_0);
                      X_7 = _mm256_add_pd(_mm256_add_pd(X_7, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += 16, Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 32 <= N_block; i += 32, X += 32, Y += (incY * 32)){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      X_4 = _mm256_loadu_pd(X + 16);
                      X_5 = _mm256_loadu_pd(X + 20);
                      X_6 = _mm256_loadu_pd(X + 24);
                      X_7 = _mm256_loadu_pd(X + 28);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      Y_4 = _mm256_set_pd(Y[(incY * 19)], Y[(incY * 18)], Y[(incY * 17)], Y[(incY * 16)]);
                      Y_5 = _mm256_set_pd(Y[(incY * 23)], Y[(incY * 22)], Y[(incY * 21)], Y[(incY * 20)]);
                      Y_6 = _mm256_set_pd(Y[(incY * 27)], Y[(incY * 26)], Y[(incY * 25)], Y[(incY * 24)]);
                      Y_7 = _mm256_set_pd(Y[(incY * 31)], Y[(incY * 30)], Y[(incY * 29)], Y[(incY * 28)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      X_2 = _mm256_loadu_pd(X + 8);
                      X_3 = _mm256_loadu_pd(X + 12);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += 16, Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      X_1 = _mm256_loadu_pd(X + 4);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_loadu_pd(X);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[2]:0, (N_block - i)>1?X[1]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32), Y += 32){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      X_4 = _mm256_set_pd(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                      X_5 = _mm256_set_pd(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                      X_6 = _mm256_set_pd(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                      X_7 = _mm256_set_pd(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      Y_4 = _mm256_loadu_pd(Y + 16);
                      Y_5 = _mm256_loadu_pd(Y + 20);
                      Y_6 = _mm256_loadu_pd(Y + 24);
                      Y_7 = _mm256_loadu_pd(Y + 28);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_4, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_4 = _mm256_add_pd(_mm256_add_pd(X_4, q_0), q_0);
                      X_5 = _mm256_add_pd(_mm256_add_pd(X_5, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_6, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_6 = _mm256_add_pd(_mm256_add_pd(X_6, q_0), q_0);
                      X_7 = _mm256_add_pd(_mm256_add_pd(X_7, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32), Y += 32){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      X_4 = _mm256_set_pd(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                      X_5 = _mm256_set_pd(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                      X_6 = _mm256_set_pd(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                      X_7 = _mm256_set_pd(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      Y_4 = _mm256_loadu_pd(Y + 16);
                      Y_5 = _mm256_loadu_pd(Y + 20);
                      Y_6 = _mm256_loadu_pd(Y + 24);
                      Y_7 = _mm256_loadu_pd(Y + 28);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      Y_2 = _mm256_loadu_pd(Y + 8);
                      Y_3 = _mm256_loadu_pd(Y + 12);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_loadu_pd(Y);
                      Y_1 = _mm256_loadu_pd(Y + 4);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_loadu_pd(Y);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[2]:0, (N_block - i)>1?Y[1]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm256_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm256_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32), Y += (incY * 32)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      X_4 = _mm256_set_pd(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                      X_5 = _mm256_set_pd(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                      X_6 = _mm256_set_pd(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                      X_7 = _mm256_set_pd(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      Y_4 = _mm256_set_pd(Y[(incY * 19)], Y[(incY * 18)], Y[(incY * 17)], Y[(incY * 16)]);
                      Y_5 = _mm256_set_pd(Y[(incY * 23)], Y[(incY * 22)], Y[(incY * 21)], Y[(incY * 20)]);
                      Y_6 = _mm256_set_pd(Y[(incY * 27)], Y[(incY * 26)], Y[(incY * 25)], Y[(incY * 24)]);
                      Y_7 = _mm256_set_pd(Y[(incY * 31)], Y[(incY * 30)], Y[(incY * 29)], Y[(incY * 28)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_4, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_4 = _mm256_add_pd(_mm256_add_pd(X_4, q_0), q_0);
                      X_5 = _mm256_add_pd(_mm256_add_pd(X_5, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_6, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_6 = _mm256_add_pd(_mm256_add_pd(X_6, q_0), q_0);
                      X_7 = _mm256_add_pd(_mm256_add_pd(X_7, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_2 = _mm256_add_pd(_mm256_add_pd(X_2, q_0), q_0);
                      X_3 = _mm256_add_pd(_mm256_add_pd(X_3, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_pd(s_1, _mm256_or_pd(_mm256_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm256_mul_pd(_mm256_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      X_1 = _mm256_add_pd(_mm256_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm256_add_pd(s_0, _mm256_or_pd(_mm256_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm256_mul_pd(_mm256_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm256_add_pd(_mm256_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 32 <= N_block; i += 32, X += (incX * 32), Y += (incY * 32)){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      X_4 = _mm256_set_pd(X[(incX * 19)], X[(incX * 18)], X[(incX * 17)], X[(incX * 16)]);
                      X_5 = _mm256_set_pd(X[(incX * 23)], X[(incX * 22)], X[(incX * 21)], X[(incX * 20)]);
                      X_6 = _mm256_set_pd(X[(incX * 27)], X[(incX * 26)], X[(incX * 25)], X[(incX * 24)]);
                      X_7 = _mm256_set_pd(X[(incX * 31)], X[(incX * 30)], X[(incX * 29)], X[(incX * 28)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      Y_4 = _mm256_set_pd(Y[(incY * 19)], Y[(incY * 18)], Y[(incY * 17)], Y[(incY * 16)]);
                      Y_5 = _mm256_set_pd(Y[(incY * 23)], Y[(incY * 22)], Y[(incY * 21)], Y[(incY * 20)]);
                      Y_6 = _mm256_set_pd(Y[(incY * 27)], Y[(incY * 26)], Y[(incY * 25)], Y[(incY * 24)]);
                      Y_7 = _mm256_set_pd(Y[(incY * 31)], Y[(incY * 30)], Y[(incY * 29)], Y[(incY * 28)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);
                      X_4 = _mm256_mul_pd(X_4, Y_4);
                      X_5 = _mm256_mul_pd(X_5, Y_5);
                      X_6 = _mm256_mul_pd(X_6, Y_6);
                      X_7 = _mm256_mul_pd(X_7, Y_7);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_4, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_5, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_4 = _mm256_add_pd(X_4, q_0);
                        X_5 = _mm256_add_pd(X_5, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_4, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_5, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_6, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_7, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_6 = _mm256_add_pd(X_6, q_0);
                        X_7 = _mm256_add_pd(X_7, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_6, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      X_2 = _mm256_set_pd(X[(incX * 11)], X[(incX * 10)], X[(incX * 9)], X[(incX * 8)]);
                      X_3 = _mm256_set_pd(X[(incX * 15)], X[(incX * 14)], X[(incX * 13)], X[(incX * 12)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      Y_2 = _mm256_set_pd(Y[(incY * 11)], Y[(incY * 10)], Y[(incY * 9)], Y[(incY * 8)]);
                      Y_3 = _mm256_set_pd(Y[(incY * 15)], Y[(incY * 14)], Y[(incY * 13)], Y[(incY * 12)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);
                      X_2 = _mm256_mul_pd(X_2, Y_2);
                      X_3 = _mm256_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_2, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_2 = _mm256_add_pd(X_2, q_0);
                        X_3 = _mm256_add_pd(X_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_3, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      X_1 = _mm256_set_pd(X[(incX * 7)], X[(incX * 6)], X[(incX * 5)], X[(incX * 4)]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      Y_1 = _mm256_set_pd(Y[(incY * 7)], Y[(incY * 6)], Y[(incY * 5)], Y[(incY * 4)]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);
                      X_1 = _mm256_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm256_add_pd(s_1, _mm256_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        q_1 = _mm256_sub_pd(s_1, q_1);
                        X_0 = _mm256_add_pd(X_0, q_0);
                        X_1 = _mm256_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(X_1, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm256_set_pd(X[(incX * 3)], X[(incX * 2)], X[incX], X[0]);
                      Y_0 = _mm256_set_pd(Y[(incY * 3)], Y[(incY * 2)], Y[incY], Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i < N_block){
                      X_0 = _mm256_set_pd(0, (N_block - i)>2?X[(incX * 2)]:0, (N_block - i)>1?X[incX]:0, X[0]);
                      Y_0 = _mm256_set_pd(0, (N_block - i)>2?Y[(incY * 2)]:0, (N_block - i)>1?Y[incY]:0, Y[0]);
                      X_0 = _mm256_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        q_0 = _mm256_add_pd(s_0, _mm256_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        q_0 = _mm256_sub_pd(s_0, q_0);
                        X_0 = _mm256_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = _mm256_sub_pd(s_buffer[(j * 2)], _mm256_set_pd(priZ[(incpriZ * j)], priZ[(incpriZ * j)], priZ[(incpriZ * j)], 0));
                cons_tmp = _mm256_broadcast_sd(priZ + (incpriZ * j));
                s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_sub_pd(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm256_store_pd(cons_buffer_tmp, s_buffer[(j * 2)]);
                priZ[(incpriZ * j)] = cons_buffer_tmp[0] + cons_buffer_tmp[1] + cons_buffer_tmp[2] + cons_buffer_tmp[3];
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
              __m128d X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11;
              __m128d Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d q_0, q_1, q_2, q_3;
              __m128d s_0_0, s_0_1, s_0_2, s_0_3;
              __m128d s_1_0, s_1_1, s_1_2, s_1_3;

              s_0_0 = s_0_1 = s_0_2 = s_0_3 = _mm_load1_pd(priZ);
              s_1_0 = s_1_1 = s_1_2 = s_1_3 = _mm_load1_pd(priZ + incpriZ);

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 24 <= N_block; i += 24, X += 24, Y += 24){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      X_8 = _mm_loadu_pd(X + 16);
                      X_9 = _mm_loadu_pd(X + 18);
                      X_10 = _mm_loadu_pd(X + 20);
                      X_11 = _mm_loadu_pd(X + 22);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      Y_8 = _mm_loadu_pd(Y + 16);
                      Y_9 = _mm_loadu_pd(Y + 18);
                      Y_10 = _mm_loadu_pd(Y + 20);
                      Y_11 = _mm_loadu_pd(Y + 22);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_8, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_9, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_10, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_11, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_8 = _mm_add_pd(_mm_add_pd(X_8, q_0), q_0);
                      X_9 = _mm_add_pd(_mm_add_pd(X_9, q_1), q_1);
                      X_10 = _mm_add_pd(_mm_add_pd(X_10, q_2), q_2);
                      X_11 = _mm_add_pd(_mm_add_pd(X_11, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += 16, Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 24 <= N_block; i += 24, X += 24, Y += 24){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      X_8 = _mm_loadu_pd(X + 16);
                      X_9 = _mm_loadu_pd(X + 18);
                      X_10 = _mm_loadu_pd(X + 20);
                      X_11 = _mm_loadu_pd(X + 22);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      Y_8 = _mm_loadu_pd(Y + 16);
                      Y_9 = _mm_loadu_pd(Y + 18);
                      Y_10 = _mm_loadu_pd(Y + 20);
                      Y_11 = _mm_loadu_pd(Y + 22);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_11, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_8 = _mm_add_pd(X_8, q_0);
                      X_9 = _mm_add_pd(X_9, q_1);
                      X_10 = _mm_add_pd(X_10, q_2);
                      X_11 = _mm_add_pd(X_11, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += 16, Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 24 <= N_block; i += 24, X += 24, Y += (incY * 24)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      X_8 = _mm_loadu_pd(X + 16);
                      X_9 = _mm_loadu_pd(X + 18);
                      X_10 = _mm_loadu_pd(X + 20);
                      X_11 = _mm_loadu_pd(X + 22);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      Y_8 = _mm_set_pd(Y[(incY * 17)], Y[(incY * 16)]);
                      Y_9 = _mm_set_pd(Y[(incY * 19)], Y[(incY * 18)]);
                      Y_10 = _mm_set_pd(Y[(incY * 21)], Y[(incY * 20)]);
                      Y_11 = _mm_set_pd(Y[(incY * 23)], Y[(incY * 22)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_8, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_9, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_10, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_11, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_8 = _mm_add_pd(_mm_add_pd(X_8, q_0), q_0);
                      X_9 = _mm_add_pd(_mm_add_pd(X_9, q_1), q_1);
                      X_10 = _mm_add_pd(_mm_add_pd(X_10, q_2), q_2);
                      X_11 = _mm_add_pd(_mm_add_pd(X_11, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += 16, Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 24 <= N_block; i += 24, X += 24, Y += (incY * 24)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      X_8 = _mm_loadu_pd(X + 16);
                      X_9 = _mm_loadu_pd(X + 18);
                      X_10 = _mm_loadu_pd(X + 20);
                      X_11 = _mm_loadu_pd(X + 22);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      Y_8 = _mm_set_pd(Y[(incY * 17)], Y[(incY * 16)]);
                      Y_9 = _mm_set_pd(Y[(incY * 19)], Y[(incY * 18)]);
                      Y_10 = _mm_set_pd(Y[(incY * 21)], Y[(incY * 20)]);
                      Y_11 = _mm_set_pd(Y[(incY * 23)], Y[(incY * 22)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_11, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_8 = _mm_add_pd(X_8, q_0);
                      X_9 = _mm_add_pd(X_9, q_1);
                      X_10 = _mm_add_pd(X_10, q_2);
                      X_11 = _mm_add_pd(X_11, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += 16, Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 24 <= N_block; i += 24, X += (incX * 24), Y += 24){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      X_8 = _mm_set_pd(X[(incX * 17)], X[(incX * 16)]);
                      X_9 = _mm_set_pd(X[(incX * 19)], X[(incX * 18)]);
                      X_10 = _mm_set_pd(X[(incX * 21)], X[(incX * 20)]);
                      X_11 = _mm_set_pd(X[(incX * 23)], X[(incX * 22)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      Y_8 = _mm_loadu_pd(Y + 16);
                      Y_9 = _mm_loadu_pd(Y + 18);
                      Y_10 = _mm_loadu_pd(Y + 20);
                      Y_11 = _mm_loadu_pd(Y + 22);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_8, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_9, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_10, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_11, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_8 = _mm_add_pd(_mm_add_pd(X_8, q_0), q_0);
                      X_9 = _mm_add_pd(_mm_add_pd(X_9, q_1), q_1);
                      X_10 = _mm_add_pd(_mm_add_pd(X_10, q_2), q_2);
                      X_11 = _mm_add_pd(_mm_add_pd(X_11, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 24 <= N_block; i += 24, X += (incX * 24), Y += 24){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      X_8 = _mm_set_pd(X[(incX * 17)], X[(incX * 16)]);
                      X_9 = _mm_set_pd(X[(incX * 19)], X[(incX * 18)]);
                      X_10 = _mm_set_pd(X[(incX * 21)], X[(incX * 20)]);
                      X_11 = _mm_set_pd(X[(incX * 23)], X[(incX * 22)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      Y_8 = _mm_loadu_pd(Y + 16);
                      Y_9 = _mm_loadu_pd(Y + 18);
                      Y_10 = _mm_loadu_pd(Y + 20);
                      Y_11 = _mm_loadu_pd(Y + 22);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_11, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_8 = _mm_add_pd(X_8, q_0);
                      X_9 = _mm_add_pd(X_9, q_1);
                      X_10 = _mm_add_pd(X_10, q_2);
                      X_11 = _mm_add_pd(X_11, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += 16;
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 24 <= N_block; i += 24, X += (incX * 24), Y += (incY * 24)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      X_8 = _mm_set_pd(X[(incX * 17)], X[(incX * 16)]);
                      X_9 = _mm_set_pd(X[(incX * 19)], X[(incX * 18)]);
                      X_10 = _mm_set_pd(X[(incX * 21)], X[(incX * 20)]);
                      X_11 = _mm_set_pd(X[(incX * 23)], X[(incX * 22)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      Y_8 = _mm_set_pd(Y[(incY * 17)], Y[(incY * 16)]);
                      Y_9 = _mm_set_pd(Y[(incY * 19)], Y[(incY * 18)]);
                      Y_10 = _mm_set_pd(Y[(incY * 21)], Y[(incY * 20)]);
                      Y_11 = _mm_set_pd(Y[(incY * 23)], Y[(incY * 22)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_8, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_9, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_10, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_11, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_8 = _mm_add_pd(_mm_add_pd(X_8, q_0), q_0);
                      X_9 = _mm_add_pd(_mm_add_pd(X_9, q_1), q_1);
                      X_10 = _mm_add_pd(_mm_add_pd(X_10, q_2), q_2);
                      X_11 = _mm_add_pd(_mm_add_pd(X_11, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_2), q_2);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(q_2, s_0_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(q_3, s_0_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 24 <= N_block; i += 24, X += (incX * 24), Y += (incY * 24)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      X_8 = _mm_set_pd(X[(incX * 17)], X[(incX * 16)]);
                      X_9 = _mm_set_pd(X[(incX * 19)], X[(incX * 18)]);
                      X_10 = _mm_set_pd(X[(incX * 21)], X[(incX * 20)]);
                      X_11 = _mm_set_pd(X[(incX * 23)], X[(incX * 22)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      Y_8 = _mm_set_pd(Y[(incY * 17)], Y[(incY * 16)]);
                      Y_9 = _mm_set_pd(Y[(incY * 19)], Y[(incY * 18)]);
                      Y_10 = _mm_set_pd(Y[(incY * 21)], Y[(incY * 20)]);
                      Y_11 = _mm_set_pd(Y[(incY * 23)], Y[(incY * 22)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);
                      X_8 = _mm_mul_pd(X_8, Y_8);
                      X_9 = _mm_mul_pd(X_9, Y_9);
                      X_10 = _mm_mul_pd(X_10, Y_10);
                      X_11 = _mm_mul_pd(X_11, Y_11);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_11, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_8 = _mm_add_pd(X_8, q_0);
                      X_9 = _mm_add_pd(X_9, q_1);
                      X_10 = _mm_add_pd(X_10, q_2);
                      X_11 = _mm_add_pd(X_11, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_8, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_9, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_10, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_11, blp_mask_tmp));
                    }
                    if(i + 16 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      X_6 = _mm_add_pd(X_6, q_2);
                      X_7 = _mm_add_pd(X_7, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_7, blp_mask_tmp));
                      i += 16, X += (incX * 16), Y += (incY * 16);
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_0_2 = _mm_add_pd(s_0_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_3 = _mm_add_pd(s_0_3, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      q_2 = _mm_sub_pd(q_2, s_0_2);
                      q_3 = _mm_sub_pd(q_3, s_0_3);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      X_2 = _mm_add_pd(X_2, q_2);
                      X_3 = _mm_add_pd(X_3, q_3);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      s_1_2 = _mm_add_pd(s_1_2, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_3 = _mm_add_pd(s_1_3, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_pd(s_0_0, _mm_set_pd(priZ[0], 0));
              cons_tmp = _mm_load1_pd(priZ);
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_2, cons_tmp));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_3, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_1_0 = _mm_sub_pd(s_1_0, _mm_set_pd(priZ[incpriZ], 0));
              cons_tmp = _mm_load1_pd(priZ + incpriZ);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_2, cons_tmp));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_3, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128d X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7;
              __m128d Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm_load1_pd(priZ);
              s_1_0 = s_1_1 = _mm_load1_pd(priZ + incpriZ);
              s_2_0 = s_2_1 = _mm_load1_pd(priZ + (incpriZ * 2));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_0), q_0);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += 16){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_0), q_0);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += 16, Y += (incY * 16)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      X_6 = _mm_loadu_pd(X + 12);
                      X_7 = _mm_loadu_pd(X + 14);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_0), q_0);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += 16){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      Y_6 = _mm_loadu_pd(Y + 12);
                      Y_7 = _mm_loadu_pd(Y + 14);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_7, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_6 = _mm_add_pd(_mm_add_pd(X_6, q_0), q_0);
                      X_7 = _mm_add_pd(_mm_add_pd(X_7, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, X += (incX * 16), Y += (incY * 16)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      X_6 = _mm_set_pd(X[(incX * 13)], X[(incX * 12)]);
                      X_7 = _mm_set_pd(X[(incX * 15)], X[(incX * 14)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      Y_6 = _mm_set_pd(Y[(incY * 13)], Y[(incY * 12)]);
                      Y_7 = _mm_set_pd(Y[(incY * 15)], Y[(incY * 14)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);
                      X_6 = _mm_mul_pd(X_6, Y_6);
                      X_7 = _mm_mul_pd(X_7, Y_7);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_7, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_6 = _mm_add_pd(X_6, q_0);
                      X_7 = _mm_add_pd(X_7, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_6, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_pd(s_0_0, _mm_set_pd(priZ[0], 0));
              cons_tmp = _mm_load1_pd(priZ);
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_1_0 = _mm_sub_pd(s_1_0, _mm_set_pd(priZ[incpriZ], 0));
              cons_tmp = _mm_load1_pd(priZ + incpriZ);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_2_0 = _mm_sub_pd(s_2_0, _mm_set_pd(priZ[(incpriZ * 2)], 0));
              cons_tmp = _mm_load1_pd(priZ + (incpriZ * 2));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              priZ[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128d X_0, X_1, X_2, X_3, X_4, X_5;
              __m128d Y_0, Y_1, Y_2, Y_3, Y_4, Y_5;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;
              __m128d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm_load1_pd(priZ);
              s_1_0 = s_1_1 = _mm_load1_pd(priZ + incpriZ);
              s_2_0 = s_2_1 = _mm_load1_pd(priZ + (incpriZ * 2));
              s_3_0 = s_3_1 = _mm_load1_pd(priZ + (incpriZ * 3));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 12 <= N_block; i += 12, X += 12, Y += 12){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 12 <= N_block; i += 12, X += 12, Y += 12){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 12 <= N_block; i += 12, X += 12, Y += (incY * 12)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 12 <= N_block; i += 12, X += 12, Y += (incY * 12)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      X_4 = _mm_loadu_pd(X + 8);
                      X_5 = _mm_loadu_pd(X + 10);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += 8, Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 12 <= N_block; i += 12, X += (incX * 12), Y += 12){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 12 <= N_block; i += 12, X += (incX * 12), Y += 12){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      Y_4 = _mm_loadu_pd(Y + 8);
                      Y_5 = _mm_loadu_pd(Y + 10);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += 8;
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 12 <= N_block; i += 12, X += (incX * 12), Y += (incY * 12)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_4 = _mm_add_pd(_mm_add_pd(X_4, q_0), q_0);
                      X_5 = _mm_add_pd(_mm_add_pd(X_5, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_0), q_0);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(q_1, s_0_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_0 = _mm_mul_pd(_mm_sub_pd(q_0, s_0_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 12 <= N_block; i += 12, X += (incX * 12), Y += (incY * 12)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      X_4 = _mm_set_pd(X[(incX * 9)], X[(incX * 8)]);
                      X_5 = _mm_set_pd(X[(incX * 11)], X[(incX * 10)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      Y_4 = _mm_set_pd(Y[(incY * 9)], Y[(incY * 8)]);
                      Y_5 = _mm_set_pd(Y[(incY * 11)], Y[(incY * 10)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);
                      X_4 = _mm_mul_pd(X_4, Y_4);
                      X_5 = _mm_mul_pd(X_5, Y_5);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_4 = _mm_add_pd(X_4, q_0);
                      X_5 = _mm_add_pd(X_5, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_4, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_5, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_3, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_2 = _mm_add_pd(X_2, q_0);
                      X_3 = _mm_add_pd(X_3, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_2, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_3, blp_mask_tmp));
                      i += 8, X += (incX * 8), Y += (incY * 8);
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      q_1 = _mm_sub_pd(q_1, s_1_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(X_1, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      q_1 = _mm_sub_pd(q_1, s_2_1);
                      X_0 = _mm_add_pd(X_0, q_0);
                      X_1 = _mm_add_pd(X_1, q_1);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      s_3_1 = _mm_add_pd(s_3_1, _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      q_0 = s_0_0;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_1_0;
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_1_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      q_0 = s_2_0;
                      s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(X_0, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_2_0);
                      X_0 = _mm_add_pd(X_0, q_0);
                      s_3_0 = _mm_add_pd(s_3_0, _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_pd(s_0_0, _mm_set_pd(priZ[0], 0));
              cons_tmp = _mm_load1_pd(priZ);
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              priZ[0] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_1_0 = _mm_sub_pd(s_1_0, _mm_set_pd(priZ[incpriZ], 0));
              cons_tmp = _mm_load1_pd(priZ + incpriZ);
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              priZ[incpriZ] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_2_0 = _mm_sub_pd(s_2_0, _mm_set_pd(priZ[(incpriZ * 2)], 0));
              cons_tmp = _mm_load1_pd(priZ + (incpriZ * 2));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              priZ[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
              s_3_0 = _mm_sub_pd(s_3_0, _mm_set_pd(priZ[(incpriZ * 3)], 0));
              cons_tmp = _mm_load1_pd(priZ + (incpriZ * 3));
              s_3_0 = _mm_add_pd(s_3_0, _mm_sub_pd(s_3_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_3_0);
              priZ[(incpriZ * 3)] = cons_buffer_tmp[0] + cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128d X_0, X_1, X_2, X_3;
              __m128d Y_0, Y_1, Y_2, Y_3;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d q_0, q_1, q_2, q_3;
              __m128d s_0, s_1, s_2, s_3;
              __m128d s_buffer[(binned_DBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = _mm_load1_pd(priZ + (incpriZ * j));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 8 <= N_block; i += 8, X += 8, Y += 8){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(s_2, q_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(s_3, q_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, X += 8, Y += 8){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 8 <= N_block; i += 8, X += 8, Y += (incY * 8)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(s_2, q_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(s_3, q_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, X += 8, Y += (incY * 8)){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      X_2 = _mm_loadu_pd(X + 4);
                      X_3 = _mm_loadu_pd(X + 6);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      X_1 = _mm_loadu_pd(X + 2);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += 4, Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_loadu_pd(X);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += 2, Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (N_block - i), Y += (incY * (N_block - i));
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 8 <= N_block; i += 8, X += (incX * 8), Y += 8){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(s_2, q_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(s_3, q_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, X += (incX * 8), Y += 8){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      Y_2 = _mm_loadu_pd(Y + 4);
                      Y_3 = _mm_loadu_pd(Y + 6);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_loadu_pd(Y);
                      Y_1 = _mm_loadu_pd(Y + 2);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += 4;
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_loadu_pd(Y);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += 2;
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (N_block - i);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = _mm_set1_pd(binned_DMCOMPRESSION);
                    expansion_0 = _mm_set1_pd(binned_DMEXPANSION * 0.5);
                    for(i = 0; i + 8 <= N_block; i += 8, X += (incX * 8), Y += (incY * 8)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(X_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(X_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      q_2 = _mm_mul_pd(_mm_sub_pd(s_2, q_2), expansion_0);
                      q_3 = _mm_mul_pd(_mm_sub_pd(s_3, q_3), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      X_2 = _mm_add_pd(_mm_add_pd(X_2, q_2), q_2);
                      X_3 = _mm_add_pd(_mm_add_pd(X_3, q_3), q_3);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(X_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      q_1 = _mm_mul_pd(_mm_sub_pd(s_1, q_1), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      X_1 = _mm_add_pd(_mm_add_pd(X_1, q_1), q_1);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      s_0 = s_buffer[0];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(X_0, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      q_0 = _mm_mul_pd(_mm_sub_pd(s_0, q_0), expansion_0);
                      X_0 = _mm_add_pd(_mm_add_pd(X_0, q_0), q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, X += (incX * 8), Y += (incY * 8)){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      X_2 = _mm_set_pd(X[(incX * 5)], X[(incX * 4)]);
                      X_3 = _mm_set_pd(X[(incX * 7)], X[(incX * 6)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      Y_2 = _mm_set_pd(Y[(incY * 5)], Y[(incY * 4)]);
                      Y_3 = _mm_set_pd(Y[(incY * 7)], Y[(incY * 6)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);
                      X_2 = _mm_mul_pd(X_2, Y_2);
                      X_3 = _mm_mul_pd(X_3, Y_3);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(X_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(X_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                        X_2 = _mm_add_pd(X_2, q_2);
                        X_3 = _mm_add_pd(X_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_pd(s_buffer[((j * 4) + 2)], _mm_or_pd(X_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_pd(s_buffer[((j * 4) + 3)], _mm_or_pd(X_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      X_1 = _mm_set_pd(X[(incX * 3)], X[(incX * 2)]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      Y_1 = _mm_set_pd(Y[(incY * 3)], Y[(incY * 2)]);
                      X_0 = _mm_mul_pd(X_0, Y_0);
                      X_1 = _mm_mul_pd(X_1, Y_1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(X_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        X_0 = _mm_add_pd(X_0, q_0);
                        X_1 = _mm_add_pd(X_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_pd(s_buffer[((j * 4) + 1)], _mm_or_pd(X_1, blp_mask_tmp));
                      i += 4, X += (incX * 4), Y += (incY * 4);
                    }
                    if(i + 2 <= N_block){
                      X_0 = _mm_set_pd(X[incX], X[0]);
                      Y_0 = _mm_set_pd(Y[incY], Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      i += 2, X += (incX * 2), Y += (incY * 2);
                    }
                    if(i < N_block){
                      X_0 = _mm_set_pd(0, X[0]);
                      Y_0 = _mm_set_pd(0, Y[0]);
                      X_0 = _mm_mul_pd(X_0, Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(X_0, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        X_0 = _mm_add_pd(X_0, q_0);
                      }
                      s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_or_pd(X_0, blp_mask_tmp));
                      X += (incX * (N_block - i)), Y += (incY * (N_block - i));
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = _mm_sub_pd(s_buffer[(j * 4)], _mm_set_pd(priZ[(incpriZ * j)], 0));
                cons_tmp = _mm_load1_pd(priZ + (incpriZ * j));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 1)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 2)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_pd(s_buffer[(j * 4)], _mm_sub_pd(s_buffer[((j * 4) + 3)], cons_tmp));
                _mm_store_pd(cons_buffer_tmp, s_buffer[(j * 4)]);
                priZ[(incpriZ * j)] = cons_buffer_tmp[0] + cons_buffer_tmp[1];
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
              double X_0;
              double Y_0;
              double compression_0;
              double expansion_0;
              double q_0;
              double s_0_0;
              double s_1_0;
              double s_2_0;

              s_0_0 = priZ[0];
              s_1_0 = priZ[incpriZ];
              s_2_0 = priZ[(incpriZ * 2)];

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = ((q_0 - s_0_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = (q_0 - s_0_0);
                      X_0 = (X_0 + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = ((q_0 - s_0_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = (q_0 - s_0_0);
                      X_0 = (X_0 + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = ((q_0 - s_0_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = (q_0 - s_0_0);
                      X_0 = (X_0 + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = ((q_0 - s_0_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      q_0 = s_0_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      q_0 = (q_0 - s_0_0);
                      X_0 = (X_0 + q_0);
                      q_0 = s_1_0;
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      X_0 = (X_0 + q_0);
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                    }
                  }
                }
              }

              priZ[0] = s_0_0;
              priZ[incpriZ] = s_1_0;
              priZ[(incpriZ * 2)] = s_2_0;

            }
            break;
          default:
            {
              int i, j;
              double X_0;
              double Y_0;
              double compression_0;
              double expansion_0;
              double q_0;
              double s_0;
              double s_buffer[binned_DBMAXFOLD];

              for(j = 0; j < fold; j += 1){
                s_buffer[j] = priZ[(incpriZ * j)];
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      s_0 = s_buffer[0];
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      s_buffer[0] = q_0;
                      q_0 = ((s_0 - q_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      s_0 = s_buffer[0];
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      s_buffer[0] = q_0;
                      q_0 = ((s_0 - q_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += 1, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      s_0 = s_buffer[0];
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      s_buffer[0] = q_0;
                      q_0 = ((s_0 - q_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += 1){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ)){
                    compression_0 = binned_DMCOMPRESSION;
                    expansion_0 = binned_DMEXPANSION * 0.5;
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      s_0 = s_buffer[0];
                      blp_tmp.d = (X_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      s_buffer[0] = q_0;
                      q_0 = ((s_0 - q_0) * expansion_0);
                      X_0 = ((X_0 + q_0) + q_0);
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, X += incX, Y += incY){
                      X_0 = X[0];
                      Y_0 = Y[0];
                      X_0 = (X_0 * Y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[j];
                        blp_tmp.d = X_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        s_buffer[j] = q_0;
                        q_0 = (s_0 - q_0);
                        X_0 = (X_0 + q_0);
                      }
                      blp_tmp.d = X_0;
                      blp_tmp.l |= 1;
                      s_buffer[j] = s_buffer[j] + blp_tmp.d;
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                priZ[(incpriZ * j)] = s_buffer[j];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    deposits += N_block;
  }

  binned_dmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
}