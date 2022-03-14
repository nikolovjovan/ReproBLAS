#include <stdlib.h>
#include <math.h>

#include "../config.h"
#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
import depositDotU
import vectorizations
from src.common import blockSize
from scripts import terminal

code_block = generate.CodeBlock()
vectorizations.conditionally_include_vectorizations(code_block)
cog.out(str(code_block))

cog.outl()

cog.out(generate.generate(blockSize.BlockSize("zmzdotu", "N_block_MAX", 32, terminal.get_diendurance(), terminal.get_diendurance(), ["bench_rzdotu_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
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
 * @brief Add to manually specified binned complex double precision Z the unconjugated dot product of complex double precision vectors X and Y
 *
 * Add to Z to the binned sum of the pairwise products of X and Y.
 *
 * @param fold the fold of the binned types
 * @param N vector length
 * @param X complex double precision vector
 * @param incX X vector stride (use every incX'th element)
 * @param Y complex double precision vector
 * @param incY Y vector stride (use every incY'th element)
 * @param priZ Z's primary vector
 * @param incpriZ stride within Z's primary vector (use every incpriZ'th element)
 * @param carZ Z's carry vector
 * @param inccarZ stride within Z's carry vector (use every inccarZ'th element)
 *
 * @author Peter Ahrens
 * @date   15 Jan 2016
 */
void binnedBLAS_zmzdotu(const int fold, const int N, const void *X, const int incX, const void *Y, const int incY, double *priZ, const int incpriZ, double *carZ, const int inccarZ){
  double amaxm[2];
  int i, j;
  int N_block = N_block_MAX;
  int deposits = 0;

  const double *x = (const double*)X;
  const double *y = (const double*)Y;

  for (i = 0; i < N; i += N_block) {
    N_block = MIN((N - i), N_block);

    binnedBLAS_zamaxm_sub(N_block, x, incX, y, incY, amaxm);

    if (isinf(amaxm[0]) || isinf(priZ[0])){
      for (j = 0; j < N_block; j++){
        priZ[0] += x[j * 2 * incX] * y[j * 2 * incY] - x[j * 2 * incX + 1] * y[j * 2 * incY + 1];
      }
    }
    if (isinf(amaxm[1]) || isinf(priZ[1])){
      for (j = 0; j < N_block; j++){
        priZ[1] += x[j * 2 * incX] * y[j * 2 * incY + 1] + x[j * 2 * incX + 1] * y[j * 2 * incY];
      }
    }
    if (isnan(priZ[0]) && isnan(priZ[1])){
      return;
    } else if (isinf(priZ[0]) && isinf(priZ[1])){
      x += N_block * 2 * incX;
      y += N_block * 2 * incY;
      continue;
    }
    if (ISNANINF(priZ[0])){
      amaxm[0] = priZ[0];
    }
    if (ISNANINF(priZ[1])){
      amaxm[1] = priZ[1];
    }

    if (deposits + N_block > binned_DBENDURANCE/2) {
      binned_zmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
      deposits = 0;
    }

    binned_zmzupdate(fold, amaxm, priZ, incpriZ, carZ, inccarZ);

    /*[[[cog
    cog.out(generate.generate(depositDotU.DepositDotU(dataTypes.DoubleComplex, "fold", "N_block", "x", "incX", "priZ", "incpriZ", "y", "incY"), cog.inFile, args, params, mode))
    ]]]*/
    {
      #if (defined(__AVX__) && !defined(reproBLAS_no__AVX__))
        __m256d nconj_mask_tmp;
        {
          __m256d tmp;
          tmp = _mm256_set_pd(0, 1, 0, 1);
          nconj_mask_tmp = _mm256_set_pd(0, -1, 0, -1);
          nconj_mask_tmp = _mm256_xor_pd(nconj_mask_tmp, tmp);
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
              __m256d x_0, x_1, x_2, x_3, x_4, x_5;
              __m256d y_0, y_1, y_2;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 2)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += 12){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += 12){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += (incY * 12)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += 12, y += (incY * 12)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += 12){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += 12){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += (incY * 12)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 6 <= N_block; i += 6, x += (incX * 12), y += (incY * 12)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priZ)[1], ((double*)priZ)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 2) + 1)], ((double*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3;
              __m256d y_0, y_1;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 4)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priZ)[1], ((double*)priZ)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 2) + 1)], ((double*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 4) + 1)], ((double*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m256d x_0, x_1, x_2, x_3;
              __m256d y_0, y_1;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0_0, s_0_1;
              __m256d s_1_0, s_1_1;
              __m256d s_2_0, s_2_1;
              __m256d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ)));
              s_1_0 = s_1_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 2)));
              s_2_0 = s_2_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 4)));
              s_3_0 = s_3_1 = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * 6)));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(((double*)priZ)[1], ((double*)priZ)[0], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ))));
              s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[1] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 2) + 1)], ((double*)priZ)[(incpriZ * 2)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 2)));
              s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 4) + 1)], ((double*)priZ)[(incpriZ * 4)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 4)));
              s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              s_3_0 = _mm256_sub_pd(s_3_0, _mm256_set_pd(((double*)priZ)[((incpriZ * 6) + 1)], ((double*)priZ)[(incpriZ * 6)], 0, 0));
              cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * 6)));
              s_3_0 = _mm256_add_pd(s_3_0, _mm256_sub_pd(s_3_1, cons_tmp));
              _mm256_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)priZ)[(incpriZ * 6)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
              ((double*)priZ)[((incpriZ * 6) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m256d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m256d y_0, y_1, y_2, y_3;
              __m256d compression_0;
              __m256d expansion_0;
              __m256d expansion_mask_0;
              __m256d q_0, q_1;
              __m256d s_0, s_1;
              __m256d s_buffer[(binned_DBMAXFOLD * 2)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm256_broadcast_pd((__m128d *)(((double*)priZ) + (incpriZ * j * 2)));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      x_3 = _mm256_loadu_pd(((double*)x) + 12);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      y_3 = _mm256_loadu_pd(((double*)y) + 12);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += 16){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      x_3 = _mm256_loadu_pd(((double*)x) + 12);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      y_3 = _mm256_loadu_pd(((double*)y) + 12);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      x_3 = _mm256_loadu_pd(((double*)x) + 12);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      y_3 = _mm256_set_pd(((double*)y)[((incY * 14) + 1)], ((double*)y)[(incY * 14)], ((double*)y)[((incY * 12) + 1)], ((double*)y)[(incY * 12)]);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += 16, y += (incY * 16)){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      x_2 = _mm256_loadu_pd(((double*)x) + 8);
                      x_3 = _mm256_loadu_pd(((double*)x) + 12);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      y_3 = _mm256_set_pd(((double*)y)[((incY * 14) + 1)], ((double*)y)[(incY * 14)], ((double*)y)[((incY * 12) + 1)], ((double*)y)[(incY * 12)]);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      x_1 = _mm256_loadu_pd(((double*)x) + 4);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += 8, y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_loadu_pd(((double*)x));
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += ((N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      y_3 = _mm256_loadu_pd(((double*)y) + 12);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += 16){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      y_2 = _mm256_loadu_pd(((double*)y) + 8);
                      y_3 = _mm256_loadu_pd(((double*)y) + 12);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      y_1 = _mm256_loadu_pd(((double*)y) + 4);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += 8;
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_loadu_pd(((double*)y));
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += ((N_block - i) * 2);
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      y_3 = _mm256_set_pd(((double*)y)[((incY * 14) + 1)], ((double*)y)[(incY * 14)], ((double*)y)[((incY * 12) + 1)], ((double*)y)[(incY * 12)]);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }else{
                    for(i = 0; i + 8 <= N_block; i += 8, x += (incX * 16), y += (incY * 16)){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      x_2 = _mm256_set_pd(((double*)x)[((incX * 10) + 1)], ((double*)x)[(incX * 10)], ((double*)x)[((incX * 8) + 1)], ((double*)x)[(incX * 8)]);
                      x_3 = _mm256_set_pd(((double*)x)[((incX * 14) + 1)], ((double*)x)[(incX * 14)], ((double*)x)[((incX * 12) + 1)], ((double*)x)[(incX * 12)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      y_2 = _mm256_set_pd(((double*)y)[((incY * 10) + 1)], ((double*)y)[(incY * 10)], ((double*)y)[((incY * 8) + 1)], ((double*)y)[(incY * 8)]);
                      y_3 = _mm256_set_pd(((double*)y)[((incY * 14) + 1)], ((double*)y)[(incY * 14)], ((double*)y)[((incY * 12) + 1)], ((double*)y)[(incY * 12)]);
                      x_4 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_5 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_6 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_2, 0x5), _mm256_permute_pd(y_2, 0xF)), nconj_mask_tmp);
                      x_7 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_3, 0x5), _mm256_permute_pd(y_3, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));
                      x_2 = _mm256_mul_pd(x_2, _mm256_permute_pd(y_2, 0x0));
                      x_3 = _mm256_mul_pd(x_3, _mm256_permute_pd(y_3, 0x0));

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
                    }
                    if(i + 4 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      x_1 = _mm256_set_pd(((double*)x)[((incX * 6) + 1)], ((double*)x)[(incX * 6)], ((double*)x)[((incX * 4) + 1)], ((double*)x)[(incX * 4)]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      y_1 = _mm256_set_pd(((double*)y)[((incY * 6) + 1)], ((double*)y)[(incY * 6)], ((double*)y)[((incY * 4) + 1)], ((double*)y)[(incY * 4)]);
                      x_2 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_3 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_1, 0x5), _mm256_permute_pd(y_1, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));
                      x_1 = _mm256_mul_pd(x_1, _mm256_permute_pd(y_1, 0x0));

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
                      i += 4, x += (incX * 8), y += (incY * 8);
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm256_set_pd(((double*)x)[((incX * 2) + 1)], ((double*)x)[(incX * 2)], ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(((double*)y)[((incY * 2) + 1)], ((double*)y)[(incY * 2)], ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i < N_block){
                      x_0 = _mm256_set_pd(0, 0, ((double*)x)[1], ((double*)x)[0]);
                      y_0 = _mm256_set_pd(0, 0, ((double*)y)[1], ((double*)y)[0]);
                      x_1 = _mm256_xor_pd(_mm256_mul_pd(_mm256_permute_pd(x_0, 0x5), _mm256_permute_pd(y_0, 0xF)), nconj_mask_tmp);
                      x_0 = _mm256_mul_pd(x_0, _mm256_permute_pd(y_0, 0x0));

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
                      x += (incX * (N_block - i) * 2), y += (incY * (N_block - i) * 2);
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 2)] = _mm256_sub_pd(s_buffer[(j * 2)], _mm256_set_pd(((double*)priZ)[((incpriZ * j * 2) + 1)], ((double*)priZ)[(incpriZ * j * 2)], 0, 0));
                cons_tmp = _mm256_broadcast_pd((__m128d *)(((double*)((double*)priZ)) + (incpriZ * j * 2)));
                s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_sub_pd(s_buffer[((j * 2) + 1)], cons_tmp));
                _mm256_store_pd(cons_buffer_tmp, s_buffer[(j * 2)]);
                ((double*)priZ)[(incpriZ * j * 2)] = cons_buffer_tmp[0] + cons_buffer_tmp[2];
                ((double*)priZ)[((incpriZ * j * 2) + 1)] = cons_buffer_tmp[1] + cons_buffer_tmp[3];
              }

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
        }

      #elif (defined(__SSE2__) && !defined(reproBLAS_no__SSE2__))
        __m128d nconj_mask_tmp;
        {
          __m128d tmp;
          tmp = _mm_set_pd(0, 1);
          nconj_mask_tmp = _mm_set_pd(0, -1);
          nconj_mask_tmp = _mm_xor_pd(nconj_mask_tmp, tmp);
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
              __m128d x_0, x_1, x_2, x_3, x_4, x_5;
              __m128d y_0, y_1, y_2;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priZ));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 2));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(x_4, q_0);
                      x_5 = _mm_add_pd(x_5, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(x_4, q_0);
                      x_5 = _mm_add_pd(x_5, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(x_4, q_0);
                      x_5 = _mm_add_pd(x_5, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(x_5, blp_mask_tmp));
                      q_0 = _mm_sub_pd(q_0, s_0_0);
                      q_1 = _mm_sub_pd(q_1, s_0_1);
                      x_4 = _mm_add_pd(x_4, q_0);
                      x_5 = _mm_add_pd(x_5, q_1);
                      s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(x_4, blp_mask_tmp));
                      s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(x_5, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0];
              ((double*)priZ)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 3:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5;
              __m128d y_0, y_1, y_2;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priZ));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 2));
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 4));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0];
              ((double*)priZ)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 4));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          case 4:
            {
              int i;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5;
              __m128d y_0, y_1, y_2;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1;
              __m128d s_0_0, s_0_1;
              __m128d s_1_0, s_1_1;
              __m128d s_2_0, s_2_1;
              __m128d s_3_0, s_3_1;

              s_0_0 = s_0_1 = _mm_loadu_pd(((double*)priZ));
              s_1_0 = s_1_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 2));
              s_2_0 = s_2_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 4));
              s_3_0 = s_3_1 = _mm_loadu_pd(((double*)priZ) + (incpriZ * 6));

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += 6, y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += 6){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 3 <= N_block; i += 3, x += (incX * 6), y += (incY * 6)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));

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
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }
                }
              }

              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)));
              s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_0_0);
              ((double*)priZ)[0] = cons_buffer_tmp[0];
              ((double*)priZ)[1] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 2));
              s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_1_0);
              ((double*)priZ)[(incpriZ * 2)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 2) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 4));
              s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_2_0);
              ((double*)priZ)[(incpriZ * 4)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 4) + 1)] = cons_buffer_tmp[1];
              cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * 6));
              s_3_0 = _mm_add_pd(s_3_0, _mm_sub_pd(s_3_1, cons_tmp));
              _mm_store_pd(cons_buffer_tmp, s_3_0);
              ((double*)priZ)[(incpriZ * 6)] = cons_buffer_tmp[0];
              ((double*)priZ)[((incpriZ * 6) + 1)] = cons_buffer_tmp[1];

              if(SIMD_daz_ftz_new_tmp != SIMD_daz_ftz_old_tmp){
                _mm_setcsr(SIMD_daz_ftz_old_tmp);
              }
            }
            break;
          default:
            {
              int i, j;
              __m128d x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7;
              __m128d y_0, y_1, y_2, y_3;
              __m128d compression_0;
              __m128d expansion_0;
              __m128d expansion_mask_0;
              __m128d q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7;
              __m128d s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7;
              __m128d s_buffer[(binned_DBMAXFOLD * 8)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 8)] = s_buffer[((j * 8) + 1)] = s_buffer[((j * 8) + 2)] = s_buffer[((j * 8) + 3)] = s_buffer[((j * 8) + 4)] = s_buffer[((j * 8) + 5)] = s_buffer[((j * 8) + 6)] = s_buffer[((j * 8) + 7)] = _mm_loadu_pd(((double*)priZ) + (incpriZ * j * 2));
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      x_3 = _mm_loadu_pd(((double*)x) + 6);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      y_3 = _mm_loadu_pd(((double*)y) + 6);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      s_4 = s_buffer[4];
                      s_5 = s_buffer[5];
                      s_6 = s_buffer[6];
                      s_7 = s_buffer[7];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                      q_4 = _mm_add_pd(s_4, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      q_5 = _mm_add_pd(s_5, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_6 = _mm_add_pd(s_6, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                      q_7 = _mm_add_pd(s_7, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      s_buffer[4] = q_4;
                      s_buffer[5] = q_5;
                      s_buffer[6] = q_6;
                      s_buffer[7] = q_7;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      q_4 = _mm_sub_pd(s_4, q_4);
                      q_5 = _mm_sub_pd(s_5, q_5);
                      q_6 = _mm_sub_pd(s_6, q_6);
                      q_7 = _mm_sub_pd(s_7, q_7);
                      x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                      x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += 2, y += 2;
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += 8){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      x_3 = _mm_loadu_pd(((double*)x) + 6);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      y_3 = _mm_loadu_pd(((double*)y) + 6);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += 4, y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += 2, y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      x_3 = _mm_loadu_pd(((double*)x) + 6);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      y_3 = _mm_loadu_pd(((double*)y) + (incY * 6));
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      s_4 = s_buffer[4];
                      s_5 = s_buffer[5];
                      s_6 = s_buffer[6];
                      s_7 = s_buffer[7];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                      q_4 = _mm_add_pd(s_4, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      q_5 = _mm_add_pd(s_5, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_6 = _mm_add_pd(s_6, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                      q_7 = _mm_add_pd(s_7, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      s_buffer[4] = q_4;
                      s_buffer[5] = q_5;
                      s_buffer[6] = q_6;
                      s_buffer[7] = q_7;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      q_4 = _mm_sub_pd(s_4, q_4);
                      q_5 = _mm_sub_pd(s_5, q_5);
                      q_6 = _mm_sub_pd(s_6, q_6);
                      q_7 = _mm_sub_pd(s_7, q_7);
                      x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                      x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += 8, y += (incY * 8)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      x_2 = _mm_loadu_pd(((double*)x) + 4);
                      x_3 = _mm_loadu_pd(((double*)x) + 6);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      y_3 = _mm_loadu_pd(((double*)y) + (incY * 6));
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + 2);
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += 4, y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += 2, y += (incY * 2);
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      y_3 = _mm_loadu_pd(((double*)y) + 6);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      s_4 = s_buffer[4];
                      s_5 = s_buffer[5];
                      s_6 = s_buffer[6];
                      s_7 = s_buffer[7];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                      q_4 = _mm_add_pd(s_4, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      q_5 = _mm_add_pd(s_5, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_6 = _mm_add_pd(s_6, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                      q_7 = _mm_add_pd(s_7, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      s_buffer[4] = q_4;
                      s_buffer[5] = q_5;
                      s_buffer[6] = q_6;
                      s_buffer[7] = q_7;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      q_4 = _mm_sub_pd(s_4, q_4);
                      q_5 = _mm_sub_pd(s_5, q_5);
                      q_6 = _mm_sub_pd(s_6, q_6);
                      q_7 = _mm_sub_pd(s_7, q_7);
                      x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                      x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += 8){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      y_2 = _mm_loadu_pd(((double*)y) + 4);
                      y_3 = _mm_loadu_pd(((double*)y) + 6);
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + 2);
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += 4;
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += (incX * 2), y += 2;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      y_3 = _mm_loadu_pd(((double*)y) + (incY * 6));
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      s_4 = s_buffer[4];
                      s_5 = s_buffer[5];
                      s_6 = s_buffer[6];
                      s_7 = s_buffer[7];
                      q_0 = _mm_add_pd(s_0, _mm_or_pd(_mm_mul_pd(x_0, compression_0), blp_mask_tmp));
                      q_1 = _mm_add_pd(s_1, _mm_or_pd(_mm_mul_pd(x_1, compression_0), blp_mask_tmp));
                      q_2 = _mm_add_pd(s_2, _mm_or_pd(_mm_mul_pd(x_2, compression_0), blp_mask_tmp));
                      q_3 = _mm_add_pd(s_3, _mm_or_pd(_mm_mul_pd(x_3, compression_0), blp_mask_tmp));
                      q_4 = _mm_add_pd(s_4, _mm_or_pd(_mm_mul_pd(x_4, compression_0), blp_mask_tmp));
                      q_5 = _mm_add_pd(s_5, _mm_or_pd(_mm_mul_pd(x_5, compression_0), blp_mask_tmp));
                      q_6 = _mm_add_pd(s_6, _mm_or_pd(_mm_mul_pd(x_6, compression_0), blp_mask_tmp));
                      q_7 = _mm_add_pd(s_7, _mm_or_pd(_mm_mul_pd(x_7, compression_0), blp_mask_tmp));
                      s_buffer[0] = q_0;
                      s_buffer[1] = q_1;
                      s_buffer[2] = q_2;
                      s_buffer[3] = q_3;
                      s_buffer[4] = q_4;
                      s_buffer[5] = q_5;
                      s_buffer[6] = q_6;
                      s_buffer[7] = q_7;
                      q_0 = _mm_sub_pd(s_0, q_0);
                      q_1 = _mm_sub_pd(s_1, q_1);
                      q_2 = _mm_sub_pd(s_2, q_2);
                      q_3 = _mm_sub_pd(s_3, q_3);
                      q_4 = _mm_sub_pd(s_4, q_4);
                      q_5 = _mm_sub_pd(s_5, q_5);
                      q_6 = _mm_sub_pd(s_6, q_6);
                      q_7 = _mm_sub_pd(s_7, q_7);
                      x_0 = _mm_add_pd(_mm_add_pd(x_0, _mm_mul_pd(q_0, expansion_0)), _mm_mul_pd(q_0, expansion_mask_0));
                      x_1 = _mm_add_pd(_mm_add_pd(x_1, _mm_mul_pd(q_1, expansion_0)), _mm_mul_pd(q_1, expansion_mask_0));
                      x_2 = _mm_add_pd(_mm_add_pd(x_2, _mm_mul_pd(q_2, expansion_0)), _mm_mul_pd(q_2, expansion_mask_0));
                      x_3 = _mm_add_pd(_mm_add_pd(x_3, _mm_mul_pd(q_3, expansion_0)), _mm_mul_pd(q_3, expansion_mask_0));
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

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
                      x_4 = _mm_add_pd(_mm_add_pd(x_4, _mm_mul_pd(q_4, expansion_0)), _mm_mul_pd(q_4, expansion_mask_0));
                      x_5 = _mm_add_pd(_mm_add_pd(x_5, _mm_mul_pd(q_5, expansion_0)), _mm_mul_pd(q_5, expansion_mask_0));
                      x_6 = _mm_add_pd(_mm_add_pd(x_6, _mm_mul_pd(q_6, expansion_0)), _mm_mul_pd(q_6, expansion_mask_0));
                      x_7 = _mm_add_pd(_mm_add_pd(x_7, _mm_mul_pd(q_7, expansion_0)), _mm_mul_pd(q_7, expansion_mask_0));
                      for(j = 1; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }else{
                    for(i = 0; i + 4 <= N_block; i += 4, x += (incX * 8), y += (incY * 8)){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      x_2 = _mm_loadu_pd(((double*)x) + (incX * 4));
                      x_3 = _mm_loadu_pd(((double*)x) + (incX * 6));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      y_2 = _mm_loadu_pd(((double*)y) + (incY * 4));
                      y_3 = _mm_loadu_pd(((double*)y) + (incY * 6));
                      x_4 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_5 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_6 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_2, x_2, 0x1), _mm_shuffle_pd(y_2, y_2, 0x3)), nconj_mask_tmp);
                      x_7 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_3, x_3, 0x1), _mm_shuffle_pd(y_3, y_3, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));
                      x_2 = _mm_mul_pd(x_2, _mm_shuffle_pd(y_2, y_2, 0x0));
                      x_3 = _mm_mul_pd(x_3, _mm_shuffle_pd(y_3, y_3, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        s_4 = s_buffer[((j * 8) + 4)];
                        s_5 = s_buffer[((j * 8) + 5)];
                        s_6 = s_buffer[((j * 8) + 6)];
                        s_7 = s_buffer[((j * 8) + 7)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        q_4 = _mm_add_pd(s_4, _mm_or_pd(x_4, blp_mask_tmp));
                        q_5 = _mm_add_pd(s_5, _mm_or_pd(x_5, blp_mask_tmp));
                        q_6 = _mm_add_pd(s_6, _mm_or_pd(x_6, blp_mask_tmp));
                        q_7 = _mm_add_pd(s_7, _mm_or_pd(x_7, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        s_buffer[((j * 8) + 4)] = q_4;
                        s_buffer[((j * 8) + 5)] = q_5;
                        s_buffer[((j * 8) + 6)] = q_6;
                        s_buffer[((j * 8) + 7)] = q_7;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        q_4 = _mm_sub_pd(s_4, q_4);
                        q_5 = _mm_sub_pd(s_5, q_5);
                        q_6 = _mm_sub_pd(s_6, q_6);
                        q_7 = _mm_sub_pd(s_7, q_7);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                        x_4 = _mm_add_pd(x_4, q_4);
                        x_5 = _mm_add_pd(x_5, q_5);
                        x_6 = _mm_add_pd(x_6, q_6);
                        x_7 = _mm_add_pd(x_7, q_7);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      s_buffer[((j * 8) + 4)] = _mm_add_pd(s_buffer[((j * 8) + 4)], _mm_or_pd(x_4, blp_mask_tmp));
                      s_buffer[((j * 8) + 5)] = _mm_add_pd(s_buffer[((j * 8) + 5)], _mm_or_pd(x_5, blp_mask_tmp));
                      s_buffer[((j * 8) + 6)] = _mm_add_pd(s_buffer[((j * 8) + 6)], _mm_or_pd(x_6, blp_mask_tmp));
                      s_buffer[((j * 8) + 7)] = _mm_add_pd(s_buffer[((j * 8) + 7)], _mm_or_pd(x_7, blp_mask_tmp));
                    }
                    if(i + 2 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      x_1 = _mm_loadu_pd(((double*)x) + (incX * 2));
                      y_0 = _mm_loadu_pd(((double*)y));
                      y_1 = _mm_loadu_pd(((double*)y) + (incY * 2));
                      x_2 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_3 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_1, x_1, 0x1), _mm_shuffle_pd(y_1, y_1, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));
                      x_1 = _mm_mul_pd(x_1, _mm_shuffle_pd(y_1, y_1, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        s_2 = s_buffer[((j * 8) + 2)];
                        s_3 = s_buffer[((j * 8) + 3)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        q_2 = _mm_add_pd(s_2, _mm_or_pd(x_2, blp_mask_tmp));
                        q_3 = _mm_add_pd(s_3, _mm_or_pd(x_3, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        s_buffer[((j * 8) + 2)] = q_2;
                        s_buffer[((j * 8) + 3)] = q_3;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        q_2 = _mm_sub_pd(s_2, q_2);
                        q_3 = _mm_sub_pd(s_3, q_3);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                        x_2 = _mm_add_pd(x_2, q_2);
                        x_3 = _mm_add_pd(x_3, q_3);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      s_buffer[((j * 8) + 2)] = _mm_add_pd(s_buffer[((j * 8) + 2)], _mm_or_pd(x_2, blp_mask_tmp));
                      s_buffer[((j * 8) + 3)] = _mm_add_pd(s_buffer[((j * 8) + 3)], _mm_or_pd(x_3, blp_mask_tmp));
                      i += 2, x += (incX * 4), y += (incY * 4);
                    }
                    if(i + 1 <= N_block){
                      x_0 = _mm_loadu_pd(((double*)x));
                      y_0 = _mm_loadu_pd(((double*)y));
                      x_1 = _mm_xor_pd(_mm_mul_pd(_mm_shuffle_pd(x_0, x_0, 0x1), _mm_shuffle_pd(y_0, y_0, 0x3)), nconj_mask_tmp);
                      x_0 = _mm_mul_pd(x_0, _mm_shuffle_pd(y_0, y_0, 0x0));

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 8)];
                        s_1 = s_buffer[((j * 8) + 1)];
                        q_0 = _mm_add_pd(s_0, _mm_or_pd(x_0, blp_mask_tmp));
                        q_1 = _mm_add_pd(s_1, _mm_or_pd(x_1, blp_mask_tmp));
                        s_buffer[(j * 8)] = q_0;
                        s_buffer[((j * 8) + 1)] = q_1;
                        q_0 = _mm_sub_pd(s_0, q_0);
                        q_1 = _mm_sub_pd(s_1, q_1);
                        x_0 = _mm_add_pd(x_0, q_0);
                        x_1 = _mm_add_pd(x_1, q_1);
                      }
                      s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_or_pd(x_0, blp_mask_tmp));
                      s_buffer[((j * 8) + 1)] = _mm_add_pd(s_buffer[((j * 8) + 1)], _mm_or_pd(x_1, blp_mask_tmp));
                      i += 1, x += (incX * 2), y += (incY * 2);
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                cons_tmp = _mm_loadu_pd(((double*)((double*)priZ)) + (incpriZ * j * 2));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 1)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 2)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 3)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 4)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 5)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 6)], cons_tmp));
                s_buffer[(j * 8)] = _mm_add_pd(s_buffer[(j * 8)], _mm_sub_pd(s_buffer[((j * 8) + 7)], cons_tmp));
                _mm_store_pd(cons_buffer_tmp, s_buffer[(j * 8)]);
                ((double*)priZ)[(incpriZ * j * 2)] = cons_buffer_tmp[0];
                ((double*)priZ)[((incpriZ * j * 2) + 1)] = cons_buffer_tmp[1];
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
              double x_0, x_1, x_2, x_3;
              double y_0, y_1;
              double compression_0, compression_1;
              double expansion_0, expansion_1;
              double expansion_mask_0, expansion_mask_1;
              double q_0, q_1, q_2, q_3;
              double s_0_0, s_0_1, s_0_2, s_0_3;
              double s_1_0, s_1_1, s_1_2, s_1_3;
              double s_2_0, s_2_1, s_2_2, s_2_3;

              s_0_0 = s_0_2 = ((double*)priZ)[0];
              s_0_1 = s_0_3 = ((double*)priZ)[1];
              s_1_0 = s_1_2 = ((double*)priZ)[(incpriZ * 2)];
              s_1_1 = s_1_3 = ((double*)priZ)[((incpriZ * 2) + 1)];
              s_2_0 = s_2_2 = ((double*)priZ)[(incpriZ * 4)];
              s_2_1 = s_2_3 = ((double*)priZ)[((incpriZ * 4) + 1)];

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      q_0 = s_0_0;
                      q_1 = s_0_1;
                      q_2 = s_0_2;
                      q_3 = s_0_3;
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_0_0 = s_0_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_0_1 = s_0_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_0_2 = s_0_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_0_3 = s_0_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_1_0 = s_1_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_1_1 = s_1_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_1_2 = s_1_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_1_3 = s_1_3 + blp_tmp.d;
                      q_0 = (q_0 - s_1_0);
                      q_1 = (q_1 - s_1_1);
                      q_2 = (q_2 - s_1_2);
                      q_3 = (q_3 - s_1_3);
                      x_0 = (x_0 + q_0);
                      x_1 = (x_1 + q_1);
                      x_2 = (x_2 + q_2);
                      x_3 = (x_3 + q_3);
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_2_0 = s_2_0 + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_2_1 = s_2_1 + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_2_2 = s_2_2 + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_2_3 = s_2_3 + blp_tmp.d;
                    }
                  }
                }
              }

              cons_tmp = ((double*)priZ)[0];
              s_0_0 = s_0_0 + (s_0_2 - cons_tmp);
              cons_tmp = ((double*)priZ)[1];
              s_0_1 = s_0_1 + (s_0_3 - cons_tmp);
              ((double*)priZ)[0] = s_0_0;
              ((double*)priZ)[1] = s_0_1;
              cons_tmp = ((double*)priZ)[(incpriZ * 2)];
              s_1_0 = s_1_0 + (s_1_2 - cons_tmp);
              cons_tmp = ((double*)priZ)[((incpriZ * 2) + 1)];
              s_1_1 = s_1_1 + (s_1_3 - cons_tmp);
              ((double*)priZ)[(incpriZ * 2)] = s_1_0;
              ((double*)priZ)[((incpriZ * 2) + 1)] = s_1_1;
              cons_tmp = ((double*)priZ)[(incpriZ * 4)];
              s_2_0 = s_2_0 + (s_2_2 - cons_tmp);
              cons_tmp = ((double*)priZ)[((incpriZ * 4) + 1)];
              s_2_1 = s_2_1 + (s_2_3 - cons_tmp);
              ((double*)priZ)[(incpriZ * 4)] = s_2_0;
              ((double*)priZ)[((incpriZ * 4) + 1)] = s_2_1;

            }
            break;
          default:
            {
              int i, j;
              double x_0, x_1, x_2, x_3;
              double y_0, y_1;
              double compression_0, compression_1;
              double expansion_0, expansion_1;
              double expansion_mask_0, expansion_mask_1;
              double q_0, q_1, q_2, q_3;
              double s_0, s_1, s_2, s_3;
              double s_buffer[(binned_DBMAXFOLD * 4)];

              for(j = 0; j < fold; j += 1){
                s_buffer[(j * 4)] = s_buffer[((j * 4) + 2)] = ((double*)priZ)[(incpriZ * j * 2)];
                s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 3)] = ((double*)priZ)[((incpriZ * j * 2) + 1)];
              }

              if(incX == 1){
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      q_2 = s_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      q_3 = s_3 + blp_tmp.d;
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
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      q_2 = s_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      q_3 = s_3 + blp_tmp.d;
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
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += 2, y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }
                }
              }else{
                if(incY == 1){
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      q_2 = s_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      q_3 = s_3 + blp_tmp.d;
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
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += 2){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }
                }else{
                  if(binned_dmindex0(priZ) || binned_dmindex0(priZ + 1)){
                    if(binned_dmindex0(priZ)){
                      if(binned_dmindex0(priZ + 1)){
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
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      s_0 = s_buffer[0];
                      s_1 = s_buffer[1];
                      s_2 = s_buffer[2];
                      s_3 = s_buffer[3];
                      blp_tmp.d = (x_0 * compression_0);
                      blp_tmp.l |= 1;
                      q_0 = s_0 + blp_tmp.d;
                      blp_tmp.d = (x_1 * compression_1);
                      blp_tmp.l |= 1;
                      q_1 = s_1 + blp_tmp.d;
                      blp_tmp.d = (x_2 * compression_0);
                      blp_tmp.l |= 1;
                      q_2 = s_2 + blp_tmp.d;
                      blp_tmp.d = (x_3 * compression_1);
                      blp_tmp.l |= 1;
                      q_3 = s_3 + blp_tmp.d;
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
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }else{
                    for(i = 0; i + 1 <= N_block; i += 1, x += (incX * 2), y += (incY * 2)){
                      x_0 = ((double*)x)[0];
                      x_1 = ((double*)x)[1];
                      y_0 = ((double*)y)[0];
                      y_1 = ((double*)y)[1];
                      x_2 = ((x_1 * y_1) * -1);
                      x_3 = (x_0 * y_1);
                      x_0 = (x_0 * y_0);
                      x_1 = (x_1 * y_0);

                      for(j = 0; j < fold - 1; j++){
                        s_0 = s_buffer[(j * 4)];
                        s_1 = s_buffer[((j * 4) + 1)];
                        s_2 = s_buffer[((j * 4) + 2)];
                        s_3 = s_buffer[((j * 4) + 3)];
                        blp_tmp.d = x_0;
                        blp_tmp.l |= 1;
                        q_0 = s_0 + blp_tmp.d;
                        blp_tmp.d = x_1;
                        blp_tmp.l |= 1;
                        q_1 = s_1 + blp_tmp.d;
                        blp_tmp.d = x_2;
                        blp_tmp.l |= 1;
                        q_2 = s_2 + blp_tmp.d;
                        blp_tmp.d = x_3;
                        blp_tmp.l |= 1;
                        q_3 = s_3 + blp_tmp.d;
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
                      blp_tmp.d = x_0;
                      blp_tmp.l |= 1;
                      s_buffer[(j * 4)] = s_buffer[(j * 4)] + blp_tmp.d;
                      blp_tmp.d = x_1;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + blp_tmp.d;
                      blp_tmp.d = x_2;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 2)] = s_buffer[((j * 4) + 2)] + blp_tmp.d;
                      blp_tmp.d = x_3;
                      blp_tmp.l |= 1;
                      s_buffer[((j * 4) + 3)] = s_buffer[((j * 4) + 3)] + blp_tmp.d;
                    }
                  }
                }
              }

              for(j = 0; j < fold; j += 1){
                cons_tmp = ((double*)priZ)[(incpriZ * j * 2)];
                s_buffer[(j * 4)] = s_buffer[(j * 4)] + (s_buffer[((j * 4) + 2)] - cons_tmp);
                cons_tmp = ((double*)priZ)[((incpriZ * j * 2) + 1)];
                s_buffer[((j * 4) + 1)] = s_buffer[((j * 4) + 1)] + (s_buffer[((j * 4) + 3)] - cons_tmp);
                ((double*)priZ)[(incpriZ * j * 2)] = s_buffer[(j * 4)];
                ((double*)priZ)[((incpriZ * j * 2) + 1)] = s_buffer[((j * 4) + 1)];
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

  binned_zmrenorm(fold, priZ, incpriZ, carZ, inccarZ);
}