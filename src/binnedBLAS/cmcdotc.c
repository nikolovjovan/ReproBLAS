#include <stdlib.h>
#include <math.h>

#include "../config.h"
#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
import depositDotC
import vectorizations
from src.common import blockSize
from scripts import terminal

code_block = generate.CodeBlock()
vectorizations.conditionally_include_vectorizations(code_block)
cog.out(str(code_block))

cog.outl()

cog.out(generate.generate(blockSize.BlockSize("cmcdotc", "N_block_MAX", 32, terminal.get_siendurance(), terminal.get_siendurance(), ["bench_rcdotc_fold_{}".format(terminal.get_sidefaultfold())]), cog.inFile, args, params, mode))
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
 * @brief Add to manually specified binned complex single precision Z the conjugated dot product of complex single precision vectors X and Y
 *
 * Add to Z the binned sum of the pairwise products of X and conjugated Y.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X complex single precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param Y complex single precision vector
 * @param incY Y vector stride (use every incY'th element)
 * @param priZ Z's primary vector
 * @param incpriZ stride within Z's primary vector (use every incpriZ'th element)
 * @param carZ Z's carry vector
 * @param inccarZ stride within Z's carry vector (use every inccarZ'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_cmcdotc(const int fold, const int N, const void *X, const int incX, const void *Y, const int incY, float *priZ, const int incpriZ, float *carZ, const int inccarZ){
  float amaxm[2];
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;

  const float *x = (const float*)X;
  const float *y = (const float*)Y;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    binnedBLAS_camaxm_sub(N_block, x, incX, y, incY, amaxm);

    if (isinf(amaxm[0]) || isinf(priZ[0])){
      for (j = 0; j < N_block; j++){
        priZ[0] += x[j * 2 * incX] * y[j * 2 * incY] + x[j * 2 * incX + 1] * y[j * 2 * incY + 1];
      }
    }
    if (isinf(amaxm[1]) || isinf(priZ[1])){
      for (j = 0; j < N_block; j++){
        priZ[1] += x[j * 2 * incX] * y[j * 2 * incY + 1] - x[j * 2 * incX + 1] * y[j * 2 * incY];
      }
    }
    if (isnan(priZ[0]) && isnan(priZ[1])){
      return;
    } else if (isinf(priZ[0]) && isinf(priZ[1])){
      x += N_block * 2 * incX;
      y += N_block * 2 * incY;
      continue;
    }
    if (ISNANINFF(priZ[0])){
      amaxm[0] = priZ[0];
    }
    if (ISNANINFF(priZ[1])){
      amaxm[1] = priZ[1];
    }

    if (deposits + N_block > binned_SBENDURANCE/2) {
      binned_cmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
      deposits = 0;
    }

    binned_cmcupdate(fold, amaxm, priZ, incpriZ, carZ, inccarZ);

    /*[[[cog
    cog.out(generate.generate(depositDotC.DepositDotC(dataTypes.FloatComplex, "fold", "N_block", "x", "incX", "priZ", "incpriZ", "y", "incY"), cog.inFile, args, params, mode))
    ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
        __m256 conj_mask_tmp;
        {
          __m256 tmp;
          tmp = _mm256_set_ps(1, 0, 1, 0, 1, 0, 1, 0);
          conj_mask_tmp = _mm256_set_ps(-1, 0, -1, 0, -1, 0, -1, 0);
          conj_mask_tmp = _mm256_xor_ps(conj_mask_tmp, tmp);
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
              __m256 x_0, x_1, x_2, x_3;
              __m256 y_0, y_1;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;

              s_0_0 = s_0_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 2)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256 y_0, y_1, y_2, y_3;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;
              __m256 s_2_0, s_2_1;

              s_0_0 = s_0_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 4)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += 32){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += 16, y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += 32){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += 16, y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += (incY * 32)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += 16, y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += (incY * 32)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += 16, y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += 32){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += (incX * 16), y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += 32){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += (incX * 16), y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += (incY * 32)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += (incX * 16), y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += (incY * 32)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      i += 8, x += (incX * 16), y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], ((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], ((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm256_add_ps(s_2_0, _mm256_sub_ps(s_2_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256 x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256 y_0, y_1, y_2, y_3;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0_0, s_0_1;
              __m256 s_1_0, s_1_1;
              __m256 s_2_0, s_2_1;
              __m256 s_3_0, s_3_1;

              s_0_0 = s_0_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 4)));
              s_3_0 = s_3_1 = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * 6)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += 32){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += 16, y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += 32){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += 16, y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += (incY * 32)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += 16, y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += 32, y += (incY * 32)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      x_2 = _mm256_loadu_ps(((float*)x) + 16);
                      x_3 = _mm256_loadu_ps(((float*)x) + 24);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += 16, y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += 32){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += (incX * 16), y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += 32){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      y_2 = _mm256_loadu_ps(((float*)y) + 16);
                      y_3 = _mm256_loadu_ps(((float*)y) + 24);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += (incX * 16), y += 16;
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += (incY * 32)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(_mm256_mul_ps(x_6, compression_0), blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(_mm256_mul_ps(x_7, compression_0), blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(_mm256_add_ps(x_6, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_7 = _mm256_add_ps(_mm256_add_ps(x_7, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += (incX * 16), y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 16 <= N_block; i += 16, x += (incX * 32), y += (incY * 32)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_2 = _mm256_set_ps(((float*)x)[((incX * 22) + 1)], ((float*)x)[(incX * 22)], ((float*)x)[((incX * 20) + 1)], ((float*)x)[(incX * 20)], ((float*)x)[((incX * 18) + 1)], ((float*)x)[(incX * 18)], ((float*)x)[((incX * 16) + 1)], ((float*)x)[(incX * 16)]);
                      x_3 = _mm256_set_ps(((float*)x)[((incX * 30) + 1)], ((float*)x)[(incX * 30)], ((float*)x)[((incX * 28) + 1)], ((float*)x)[(incX * 28)], ((float*)x)[((incX * 26) + 1)], ((float*)x)[(incX * 26)], ((float*)x)[((incX * 24) + 1)], ((float*)x)[(incX * 24)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_2 = _mm256_set_ps(((float*)y)[((incY * 22) + 1)], ((float*)y)[(incY * 22)], ((float*)y)[((incY * 20) + 1)], ((float*)y)[(incY * 20)], ((float*)y)[((incY * 18) + 1)], ((float*)y)[(incY * 18)], ((float*)y)[((incY * 16) + 1)], ((float*)y)[(incY * 16)]);
                      y_3 = _mm256_set_ps(((float*)y)[((incY * 30) + 1)], ((float*)y)[(incY * 30)], ((float*)y)[((incY * 28) + 1)], ((float*)y)[(incY * 28)], ((float*)y)[((incY * 26) + 1)], ((float*)y)[(incY * 26)], ((float*)y)[((incY * 24) + 1)], ((float*)y)[(incY * 24)]);
                      x_4 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_5 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_6 = _mm256_mul_ps(_mm256_permute_ps(x_2, 0xB1), _mm256_permute_ps(y_2, 0xF5));
                      x_7 = _mm256_mul_ps(_mm256_permute_ps(x_3, 0xB1), _mm256_permute_ps(y_3, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm256_xor_ps(_mm256_mul_ps(x_2, _mm256_permute_ps(y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm256_xor_ps(_mm256_mul_ps(x_3, _mm256_permute_ps(y_3, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_4 = _mm256_add_ps(x_4, q_0);
                      x_5 = _mm256_add_ps(x_5, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_4, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_5, blp_mask_tmp));
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm256_add_ps(s_0_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_0_1 = _mm256_add_ps(s_0_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_0_0);
                      q_1 = _mm256_sub_ps(q_1, s_0_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      s_1_0 = _mm256_add_ps(s_1_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_1_1 = _mm256_add_ps(s_1_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_1_0);
                      q_1 = _mm256_sub_ps(q_1, s_1_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_7, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_6 = _mm256_add_ps(x_6, q_0);
                      x_7 = _mm256_add_ps(x_7, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_6, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_7, blp_mask_tmp));
                    }
                    if(i + 8 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_2 = _mm256_add_ps(x_2, q_0);
                      x_3 = _mm256_add_ps(x_3, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_3, blp_mask_tmp));
                      i += 8, x += (incX * 16), y += (incY * 16);
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm256_add_ps(s_2_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm256_add_ps(s_2_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm256_sub_ps(q_0, s_2_0);
                      q_1 = _mm256_sub_ps(q_1, s_2_1);
                      x_0 = _mm256_add_ps(x_0, q_0);
                      x_1 = _mm256_add_ps(x_1, q_1);
                      s_3_0 = _mm256_add_ps(s_3_0, _mm256_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm256_add_ps(s_3_1, _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_ps(s_0_0, _mm256_set_ps(((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], ((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm256_add_ps(s_0_0, _mm256_sub_ps(s_0_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_1_0 = _mm256_sub_ps(s_1_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], ((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_ps(s_1_0, _mm256_sub_ps(s_1_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_2_0 = _mm256_sub_ps(s_2_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], ((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], ((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm256_add_ps(s_2_0, _mm256_sub_ps(s_2_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              s_3_0 = _mm256_sub_ps(s_3_0, _mm256_set_ps(((float*)priZ)[((incpriZ * 6) + 1)], ((float*)priZ)[(incpriZ * 6)], ((float*)priZ)[((incpriZ * 6) + 1)], ((float*)priZ)[(incpriZ * 6)], ((float*)priZ)[((incpriZ * 6) + 1)], ((float*)priZ)[(incpriZ * 6)], 0, 0));
              cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * 6)));
              s_3_0 = _mm256_add_ps(s_3_0, _mm256_sub_ps(s_3_1, cons_tmp));
              _mm256_store_ps(cons_buffer_tmp, s_3_0);
              ((float*)priZ)[(incpriZ * 6)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
              ((float*)priZ)[((incpriZ * 6) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256 x_0, x_1, x_2, x_3;
              __m256 y_0, y_1;
              __m256 compression_0;
              __m256 expansion_0;
              __m256 expansion_mask_0;
              __m256 q_0, q_1;
              __m256 s_0, s_1;
              __m256 s_buffer[(binned_SBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = (__m256)_mm256_broadcast_sd((double *)(((float*)priZ) + (incpriZ * j * 2)));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      x_1 = _mm256_loadu_ps(((float*)x) + 8);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_ps(((float*)x));
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[2]:0, (N_block - i)>1?((double*)((float*)x))[1]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      y_1 = _mm256_loadu_ps(((float*)y) + 8);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_loadu_ps(((float*)y));
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[2]:0, (N_block - i)>1?((double*)((float*)y))[1]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_2 = _mm256_add_ps(_mm256_add_ps(x_2, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_3 = _mm256_add_ps(_mm256_add_ps(x_3, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      q_0 = _mm256_add_ps(s_0, _mm256_or_ps(_mm256_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm256_add_ps(s_1, _mm256_or_ps(_mm256_mul_ps(x_1, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      q_0 = _mm256_sub_ps(s_0, q_0);
                      q_1 = _mm256_sub_ps(s_1, q_1);
                      x_0 = _mm256_add_ps(_mm256_add_ps(x_0, _mm256_mul_ps(q_0, expansion_0)), _mm256_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm256_add_ps(_mm256_add_ps(x_1, _mm256_mul_ps(q_1, expansion_0)), _mm256_mul_ps(q_1, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm256_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)], ((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm256_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)], ((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_2 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_3 = _mm256_mul_ps(_mm256_permute_ps(x_1, 0xB1), _mm256_permute_ps(y_1, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm256_xor_ps(_mm256_mul_ps(x_1, _mm256_permute_ps(y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_2, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_2 = _mm256_add_ps(x_2, q_0);
                        x_3 = _mm256_add_ps(x_3, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)], ((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm256_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)], ((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i < N_block){
                      x_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)x))[(incX * 2)]:0, (N_block - i)>1?((double*)((float*)x))[incX]:0, ((double*)((float*)x))[0]);
                      y_0 = (__m256)_mm256_set_pd(0, (N_block - i)>2?((double*)((float*)y))[(incY * 2)]:0, (N_block - i)>1?((double*)((float*)y))[incY]:0, ((double*)((float*)y))[0]);
                      x_1 = _mm256_mul_ps(_mm256_permute_ps(x_0, 0xB1), _mm256_permute_ps(y_0, 0xF5));
                      x_0 = _mm256_xor_ps(_mm256_mul_ps(x_0, _mm256_permute_ps(y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 2)];
                        s_1 = s_buffer[((j * 2) + 1)];
                        q_0 = _mm256_add_ps(s_0, _mm256_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm256_add_ps(s_1, _mm256_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 2)] = q_0;
                        s_buffer[((j * 2) + 1)] = q_1;
                        q_0 = _mm256_sub_ps(s_0, q_0);
                        q_1 = _mm256_sub_ps(s_1, q_1);
                        x_0 = _mm256_add_ps(x_0, q_0);
                        x_1 = _mm256_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 2) + 1)] = _mm256_add_ps(s_buffer[((j * 2) + 1)], _mm256_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = _mm256_sub_ps(s_buffer[(j * 2)], _mm256_set_ps(((float*)priZ)[((incpriZ * j * 2) + 1)], ((float*)priZ)[(incpriZ * j * 2)], ((float*)priZ)[((incpriZ * j * 2) + 1)], ((float*)priZ)[(incpriZ * j * 2)], ((float*)priZ)[((incpriZ * j * 2) + 1)], ((float*)priZ)[(incpriZ * j * 2)], 0, 0));
                cons_tmp = (__m256)_mm256_broadcast_sd((double *)(((float*)((float*)priZ)) + (incpriZ * j * 2)));
                s_buffer[(j * 2)] = _mm256_add_ps(s_buffer[(j * 2)], _mm256_sub_ps(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm256_store_ps(cons_buffer_tmp, s_buffer[(j * 2)]);
                ((float*)priZ)[(incpriZ * j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2] + cons_buffer_tmp[4] + cons_buffer_tmp[6];
                ((float*)priZ)[((incpriZ * j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3] + cons_buffer_tmp[5] + cons_buffer_tmp[7];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
        __m128 conj_mask_tmp;
        {
          __m128 tmp;
          tmp = _mm_set_ps(1, 0, 1, 0);
          conj_mask_tmp = _mm_set_ps(-1, 0, -1, 0);
          conj_mask_tmp = _mm_xor_ps(conj_mask_tmp, tmp);
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
              __m128 x_0, x_1, x_2, x_3, x_4, x_5;
              __m128 y_0, y_1, y_2;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;

              s_0_0 = s_0_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 2)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += 12){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += 12){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(x_4, q_0);
                      x_5 = _mm_add_ps(x_5, q_1);
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += (incY * 12)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += (incY * 12)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(x_4, q_0);
                      x_5 = _mm_add_ps(x_5, q_1);
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += 12){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += 12){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(x_4, q_0);
                      x_5 = _mm_add_ps(x_5, q_1);
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += (incY * 12)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(_mm_mul_ps(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(_mm_mul_ps(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(_mm_add_ps(x_4, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_5 = _mm_add_ps(_mm_add_ps(x_5, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += (incY * 12)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_ps(s_0_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_ps(s_0_1, _mm_or_ps(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_0_0);
                      q_1 = _mm_sub_ps(q_1, s_0_1);
                      x_4 = _mm_add_ps(x_4, q_0);
                      x_5 = _mm_add_ps(x_5, q_1);
                      s_1_0 = _mm_add_ps(s_1_0, _mm_or_ps(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_ps(s_1_1, _mm_or_ps(x_5, blp_mask_tmp));
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128 x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m128 y_0, y_1, y_2, y_3;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;
              __m128 s_2_0, s_2_1;

              s_0_0 = s_0_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 4)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      x_3 = _mm_loadu_ps(((float*)x) + 12);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      y_3 = _mm_loadu_ps(((float*)y) + 12);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      x_3 = _mm_loadu_ps(((float*)x) + 12);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      y_3 = _mm_loadu_ps(((float*)y) + 12);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      x_3 = _mm_loadu_ps(((float*)x) + 12);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_3 = _mm_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)]);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      x_2 = _mm_loadu_ps(((float*)x) + 8);
                      x_3 = _mm_loadu_ps(((float*)x) + 12);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_3 = _mm_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)]);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_3 = _mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      y_3 = _mm_loadu_ps(((float*)y) + 12);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_3 = _mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      y_2 = _mm_loadu_ps(((float*)y) + 8);
                      y_3 = _mm_loadu_ps(((float*)y) + 12);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_3 = _mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_3 = _mm_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)]);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      x_2 = _mm_set_ps(((float*)x)[((incX * 10) + 1)], ((float*)x)[(incX * 10)], ((float*)x)[((incX * 8) + 1)], ((float*)x)[(incX * 8)]);
                      x_3 = _mm_set_ps(((float*)x)[((incX * 14) + 1)], ((float*)x)[(incX * 14)], ((float*)x)[((incX * 12) + 1)], ((float*)x)[(incX * 12)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      y_2 = _mm_set_ps(((float*)y)[((incY * 10) + 1)], ((float*)y)[(incY * 10)], ((float*)y)[((incY * 8) + 1)], ((float*)y)[(incY * 8)]);
                      y_3 = _mm_set_ps(((float*)y)[((incY * 14) + 1)], ((float*)y)[(incY * 14)], ((float*)y)[((incY * 12) + 1)], ((float*)y)[(incY * 12)]);
                      x_4 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_5 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_6 = _mm_mul_ps(_mm_shuffle_ps(x_2, x_2, 0xB1), _mm_shuffle_ps(y_2, y_2, 0xF5));
                      x_7 = _mm_mul_ps(_mm_shuffle_ps(x_3, x_3, 0xB1), _mm_shuffle_ps(y_3, y_3, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);
                      x_2 = _mm_xor_ps(_mm_mul_ps(x_2, _mm_shuffle_ps(y_2, y_2, 0xA0)), conj_mask_tmp);
                      x_3 = _mm_xor_ps(_mm_mul_ps(x_3, _mm_shuffle_ps(y_3, y_3, 0xA0)), conj_mask_tmp);

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128 x_0, x_1, x_2, x_3;
              __m128 y_0, y_1;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1;
              __m128 s_0_0, s_0_1;
              __m128 s_1_0, s_1_1;
              __m128 s_2_0, s_2_1;
              __m128 s_3_0, s_3_1;

              s_0_0 = s_0_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ)));
              s_1_0 = s_1_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 4)));
              s_3_0 = s_3_1 = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * 6)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_3, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_2 = _mm_add_ps(x_2, q_0);
                      x_3 = _mm_add_ps(x_3, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_2, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      q_0 = s_2_0;
                      q_1 = s_2_1;
                      s_2_0 = _mm_add_ps(s_2_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_2_1 = _mm_add_ps(s_2_1, _mm_or_ps(x_1, blp_mask_tmp));
                      q_0 = _mm_sub_ps(q_0, s_2_0);
                      q_1 = _mm_sub_ps(q_1, s_2_1);
                      x_0 = _mm_add_ps(x_0, q_0);
                      x_1 = _mm_add_ps(x_1, q_1);
                      s_3_0 = _mm_add_ps(s_3_0, _mm_or_ps(x_0, blp_mask_tmp));
                      s_3_1 = _mm_add_ps(s_3_1, _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm_sub_ps(s_0_0, _mm_set_ps(((float*)priZ)[1], ((float*)priZ)[0], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ))));
              s_0_0 = _mm_add_ps(s_0_0, _mm_sub_ps(s_0_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_0_0);
              ((float*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm_sub_ps(s_1_0, _mm_set_ps(((float*)priZ)[((incpriZ * 2) + 1)], ((float*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm_add_ps(s_1_0, _mm_sub_ps(s_1_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_1_0);
              ((float*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm_sub_ps(s_2_0, _mm_set_ps(((float*)priZ)[((incpriZ * 4) + 1)], ((float*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm_add_ps(s_2_0, _mm_sub_ps(s_2_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_2_0);
              ((float*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_3_0 = _mm_sub_ps(s_3_0, _mm_set_ps(((float*)priZ)[((incpriZ * 6) + 1)], ((float*)priZ)[(incpriZ * 6)], 0, 0));
              cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * 6)));
              s_3_0 = _mm_add_ps(s_3_0, _mm_sub_ps(s_3_1, cons_tmp));
              _mm_store_ps(cons_buffer_tmp, s_3_0);
              ((float*)priZ)[(incpriZ * 6)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((float*)priZ)[((incpriZ * 6) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128 x_0, x_1, x_2, x_3;
              __m128 y_0, y_1;
              __m128 compression_0;
              __m128 expansion_0;
              __m128 expansion_mask_0;
              __m128 q_0, q_1, q_2, q_3;
              __m128 s_0, s_1, s_2, s_3;
              __m128 s_buffer[(binned_SBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 3)] = (__m128)_mm_load1_pd((double *)(((float*)priZ) + (incpriZ * j * 2)));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_ps(((float*)x));
                      x_1 = _mm_loadu_ps(((float*)x) + 4);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_ps(((float*)x));
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      y_1 = _mm_loadu_ps(((float*)y) + 4);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_loadu_ps(((float*)y));
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      q_0 = _mm_add_ps(s_0, _mm_or_ps(_mm_mul_ps(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_ps(s_1, _mm_or_ps(_mm_mul_ps(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_ps(s_2, _mm_or_ps(_mm_mul_ps(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_ps(s_3, _mm_or_ps(_mm_mul_ps(x_3, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = _mm_sub_ps(s_0, q_0);
                      q_1 = _mm_sub_ps(s_1, q_1);
                      q_2 = _mm_sub_ps(s_2, q_2);
                      q_3 = _mm_sub_ps(s_3, q_3);
                      x_0 = _mm_add_ps(_mm_add_ps(x_0, _mm_mul_ps(q_0, expansion_0)), _mm_mul_ps(q_0, expansion_mask_0));
                      x_1 = _mm_add_ps(_mm_add_ps(x_1, _mm_mul_ps(q_1, expansion_0)), _mm_mul_ps(q_1, expansion_mask_0));
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

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
                      x_2 = _mm_add_ps(_mm_add_ps(x_2, _mm_mul_ps(q_2, expansion_0)), _mm_mul_ps(q_2, expansion_mask_0));
                      x_3 = _mm_add_ps(_mm_add_ps(x_3, _mm_mul_ps(q_3, expansion_0)), _mm_mul_ps(q_3, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      x_1 = _mm_set_ps(((float*)x)[((incX * 6) + 1)], ((float*)x)[(incX * 6)], ((float*)x)[((incX * 4) + 1)], ((float*)x)[(incX * 4)]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      y_1 = _mm_set_ps(((float*)y)[((incY * 6) + 1)], ((float*)y)[(incY * 6)], ((float*)y)[((incY * 4) + 1)], ((float*)y)[(incY * 4)]);
                      x_2 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_3 = _mm_mul_ps(_mm_shuffle_ps(x_1, x_1, 0xB1), _mm_shuffle_ps(y_1, y_1, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);
                      x_1 = _mm_xor_ps(_mm_mul_ps(x_1, _mm_shuffle_ps(y_1, y_1, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        q_2 = _mm_add_ps(s_2, _mm_or_ps(x_2, blp_mask_tmp));
                        q_3 = _mm_add_ps(s_3, _mm_or_ps(x_3, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        q_2 = _mm_sub_ps(s_2, q_2);
                        q_3 = _mm_sub_ps(s_3, q_3);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                        x_2 = _mm_add_ps(x_2, q_2);
                        x_3 = _mm_add_ps(x_3, q_3);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      s_buffer[((j * 4) + 2)] = _mm_add_ps(s_buffer[((j * 4) + 2)], _mm_or_ps(x_2, blp_mask_tmp));
                      s_buffer[((j * 4) + 3)] = _mm_add_ps(s_buffer[((j * 4) + 3)], _mm_or_ps(x_3, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_set_ps(((float*)x)[((incX * 2) + 1)], ((float*)x)[(incX * 2)], ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(((float*)y)[((incY * 2) + 1)], ((float*)y)[(incY * 2)], ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm_set_ps(0, 0, ((float*)x)[1], ((float*)x)[0]);
                      y_0 = _mm_set_ps(0, 0, ((float*)y)[1], ((float*)y)[0]);
                      x_1 = _mm_mul_ps(_mm_shuffle_ps(x_0, x_0, 0xB1), _mm_shuffle_ps(y_0, y_0, 0xF5));
                      x_0 = _mm_xor_ps(_mm_mul_ps(x_0, _mm_shuffle_ps(y_0, y_0, 0xA0)), conj_mask_tmp);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        q_0 = _mm_add_ps(s_0, _mm_or_ps(x_0, blp_mask_tmp));
                        q_1 = _mm_add_ps(s_1, _mm_or_ps(x_1, blp_mask_tmp));
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        q_0 = _mm_sub_ps(s_0, q_0);
                        q_1 = _mm_sub_ps(s_1, q_1);
                        x_0 = _mm_add_ps(x_0, q_0);
                        x_1 = _mm_add_ps(x_1, q_1);
                      }
                      s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_or_ps(x_0, blp_mask_tmp));
                      s_buffer[((j * 4) + 1)] = _mm_add_ps(s_buffer[((j * 4) + 1)], _mm_or_ps(x_1, blp_mask_tmp));
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = _mm_sub_ps(s_buffer[(j * 4)], _mm_set_ps(((float*)priZ)[((incpriZ * j * 2) + 1)], ((float*)priZ)[(incpriZ * j * 2)], 0, 0));
                cons_tmp = (__m128)_mm_load1_pd((double *)(((float*)((float*)priZ)) + (incpriZ * j * 2)));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 1)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 2)], cons_tmp));
                s_buffer[(j * 4)] = _mm_add_ps(s_buffer[(j * 4)], _mm_sub_ps(s_buffer[((j * 4) + 3)], cons_tmp));
                _mm_store_ps(cons_buffer_tmp, s_buffer[(j * 4)]);
                ((float*)priZ)[(incpriZ * j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
                ((float*)priZ)[((incpriZ * j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
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
              float x_0, x_1, x_2, x_3;
              float y_0, y_1;
              float compression_0, compression_1;
              float expansion_0, expansion_1;
              float expansion_mask_0, expansion_mask_1;
              float q_0, q_1, q_2, q_3;
              float s_0_0, s_0_1, s_0_2, s_0_3;
              float s_1_0, s_1_1, s_1_2, s_1_3;
              float s_2_0, s_2_1, s_2_2, s_2_3;

              s_0_0 = s_0_2 = ((float*)priZ)[0];
              s_0_1 = s_0_3 = ((float*)priZ)[1];
              s_1_0 = s_1_2 = ((float*)priZ)[(incpriZ * 2)];
              s_1_1 = s_1_3 = ((float*)priZ)[((incpriZ * 2) + 1)];
              s_2_0 = s_2_2 = ((float*)priZ)[(incpriZ * 4)];
              s_2_1 = s_2_3 = ((float*)priZ)[((incpriZ * 4) + 1)];

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_0_0 = s_0_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_0_1 = s_0_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_0_2 = s_0_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_0_3 = s_0_3 + blp_tmp.f;
                      q_0 = (q_0 - s_0_0);
                      q_1 = (q_1 - s_0_1);
                      q_2 = (q_2 - s_0_2);
                      q_3 = (q_3 - s_0_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      q_0 = s_1_0;
                      q_1 = s_1_1;
                      q_2 = s_1_2;
                      q_3 = s_1_3;
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_1_0 = s_1_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_1_1 = s_1_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_1_2 = s_1_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_1_3 = s_1_3 + blp_tmp.f;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_2_0 = s_2_0 + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_2_1 = s_2_1 + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_2_2 = s_2_2 + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_2_3 = s_2_3 + blp_tmp.f;
                    }
                  }
                }
              }

              cons_tmp = ((float*)priZ)[0];
              s_0_0 = s_0_0 + (s_0_2 - cons_tmp);
              cons_tmp = ((float*)priZ)[1];
              s_0_1 = s_0_1 + (s_0_3 - cons_tmp);
              ((float*)priZ)[0] = s_0_0;
              ((float*)priZ)[1] = s_0_1;
              cons_tmp = ((float*)priZ)[(incpriZ * 2)];
              s_1_0 = s_1_0 + (s_1_2 - cons_tmp);
              cons_tmp = ((float*)priZ)[((incpriZ * 2) + 1)];
              s_1_1 = s_1_1 + (s_1_3 - cons_tmp);
              ((float*)priZ)[(incpriZ * 2)] = s_1_0;
              ((float*)priZ)[((incpriZ * 2) + 1)] = s_1_1;
              cons_tmp = ((float*)priZ)[(incpriZ * 4)];
              s_2_0 = s_2_0 + (s_2_2 - cons_tmp);
              cons_tmp = ((float*)priZ)[((incpriZ * 4) + 1)];
              s_2_1 = s_2_1 + (s_2_3 - cons_tmp);
              ((float*)priZ)[(incpriZ * 4)] = s_2_0;
              ((float*)priZ)[((incpriZ * 4) + 1)] = s_2_1;

            }
            break;
          default:
            {
              int i, j;
              float x_0, x_1, x_2, x_3;
              float y_0, y_1;
              float compression_0, compression_1;
              float expansion_0, expansion_1;
              float expansion_mask_0, expansion_mask_1;
              float q_0, q_1, q_2, q_3;
              float s_0, s_1, s_2, s_3;
              float s_buffer[(binned_SBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 2)] = ((float*)priZ)[(incpriZ * j * 2)];
                s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 3)] = ((float*)priZ)[((incpriZ * j * 2) + 1)];
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      q_2 = s_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      q_3 = s_3 + blp_tmp.f;
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      q_2 = (s_2 - q_2);
                      q_3 = (s_3 - q_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      q_2 = s_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      q_3 = s_3 + blp_tmp.f;
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      q_2 = (s_2 - q_2);
                      q_3 = (s_3 - q_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      q_2 = s_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      q_3 = s_3 + blp_tmp.f;
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      q_2 = (s_2 - q_2);
                      q_3 = (s_3 - q_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }
                }else{
                  if(binned_smindex0(priZ) || binned_smindex0(priZ + 1)){
                    if(binned_smindex0(priZ)){
                      if(binned_smindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.f = (x_0 * compression_0);
                      blp_tmp.i |= 1;
                      q_0 = s_0 + blp_tmp.f;
                      blp_tmp.f = (x_1 * compression_1);
                      blp_tmp.i |= 1;
                      q_1 = s_1 + blp_tmp.f;
                      blp_tmp.f = (x_2 * compression_0);
                      blp_tmp.i |= 1;
                      q_2 = s_2 + blp_tmp.f;
                      blp_tmp.f = (x_3 * compression_1);
                      blp_tmp.i |= 1;
                      q_3 = s_3 + blp_tmp.f;
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      q_0 = (s_0 - q_0);
                      q_1 = (s_1 - q_1);
                      q_2 = (s_2 - q_2);
                      q_3 = (s_3 - q_3);
                      x_0 = ((x_0 + (q_0 * expansion_0)) + (q_0 * expansion_mask_0));
                      x_1 = ((x_1 + (q_1 * expansion_1)) + (q_1 * expansion_mask_1));
                      x_2 = ((x_2 + (q_2 * expansion_0)) + (q_2 * expansion_mask_0));
                      x_3 = ((x_3 + (q_3 * expansion_1)) + (q_3 * expansion_mask_1));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((float*)x)[0];
                      x_1 = ((float*)x)[1];
                      y_0 = ((float*)y)[0];
                      y_1 = ((float*)y)[1];
                      x_2 = (x_1 * y_1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = ((x_1 * y_0) * -1);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.f = x_0;
                        blp_tmp.i |= 1;
                        q_0 = s_0 + blp_tmp.f;
                        blp_tmp.f = x_1;
                        blp_tmp.i |= 1;
                        q_1 = s_1 + blp_tmp.f;
                        blp_tmp.f = x_2;
                        blp_tmp.i |= 1;
                        q_2 = s_2 + blp_tmp.f;
                        blp_tmp.f = x_3;
                        blp_tmp.i |= 1;
                        q_3 = s_3 + blp_tmp.f;
                        s_buffer[(j * 4)] = q_0;
                        s_buffer[((j * 4) + 1)] = q_1;
                        s_buffer[((j * 4) + 2)] = q_2;
                        s_buffer[((j * 4) + 3)] = q_3;
                        q_0 = (s_0 - q_0);
                        q_1 = (s_1 - q_1);
                        q_2 = (s_2 - q_2);
                        q_3 = (s_3 - q_3);
                        x_0 = (x_0 + q_0);
                        x_1 = (x_1 + q_1);
                        x_2 = (x_2 + q_2);
                        x_3 = (x_3 + q_3);
                      }
                      blp_tmp.f = x_0;
                      blp_tmp.i |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.f;
                      blp_tmp.f = x_1;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.f;
                      blp_tmp.f = x_2;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.f;
                      blp_tmp.f = x_3;
                      blp_tmp.i |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.f;
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                cons_tmp = ((float*)priZ)[(incpriZ * j * 2)];
                s_buffer[(j * 4)] = s_buffer[(j * 4)] + (s_buffer[((j * 4) + 2)] - cons_tmp);
                cons_tmp = ((float*)priZ)[((incpriZ * j * 2) + 1)];
                s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + (s_buffer[((j * 4) + 3)] - cons_tmp);
                ((float*)priZ)[(incpriZ * j * 2)] = s_buffer[(j * 4)];
                ((float*)priZ)[((incpriZ * j * 2) + 1)] = s_buffer[((j * 4) + 1)];
              }

            }
            break;
        }

      #endif

        }
    //[[[end]]]

    if (isinf(amaxm[0])){
      priZ[0] = amaxm[0];
    }
    if (isinf(amaxm[1])){
      priZ[1] = amaxm[1];
    }

    deposits += N_block;
  }

  binned_cmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
}