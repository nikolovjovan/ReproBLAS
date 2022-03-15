#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>

#include "../common/test_opt.h"
#include "../common/test_BLAS.h"
#include "../common/test_matmat_fill_header.h"

#include "wrap_rsgemm.h"

static opt_option max_blocks;
static opt_option shuffles;
static opt_option fold;

static void corroborate_rsgemm_options_initialize(void){
  max_blocks._int.header.type       = opt_int;
  max_blocks._int.header.short_name = 'B';
  max_blocks._int.header.long_name  = "blocks";
  max_blocks._int.header.help       = "maximum number of blocks";
  max_blocks._int.required          = 0;
  max_blocks._int.min               = 1;
  max_blocks._int.max               = INT_MAX;
  max_blocks._int.value             = 1024;

  shuffles._int.header.type       = opt_int;
  shuffles._int.header.short_name = 'S';
  shuffles._int.header.long_name  = "shuffles";
  shuffles._int.header.help       = "number of times to shuffle";
  shuffles._int.required          = 0;
  shuffles._int.min               = 0;
  shuffles._int.max               = INT_MAX;
  shuffles._int.value             = 5;

  fold._int.header.type       = opt_int;
  fold._int.header.short_name = 'k';
  fold._int.header.long_name  = "fold";
  fold._int.header.help       = "fold";
  fold._int.required          = 0;
  fold._int.min               = 2;
  fold._int.max               = binned_SBMAXFOLD;
  fold._int.value             = SIDEFAULTFOLD;
}

int corroborate_rsgemm(int fold, char Order, char TransA, char TransB, int M, int N, int K, float alpha, float *A, int lda, float* B, int ldb, float beta, float *C, float_binned *CI, int ldc, float *ref, int max_num_blocks) {

  int i;
  int j;
  int k;
  int num_blocks = 1;
  int block_K;

  float *res;
  float_binned *Ires;
  float *tmpA;
  float *tmpB;
  int CNM;

  switch(Order){
    case 'r':
    case 'R':
      CNM = M * ldc;
      break;
    default:
      CNM = ldc * N;
      break;
  }
  res = malloc(CNM * sizeof(float));
  Ires = malloc(CNM * binned_sbsize(fold));

  num_blocks = 1;
  while (num_blocks < K && num_blocks <= max_num_blocks) {
    memcpy(res, C, CNM * sizeof(float));
    memcpy(Ires, CI, CNM * binned_sbsize(fold));
    if (num_blocks == 1){
      wrap_rsgemm(fold, Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, res, ldc);
    }else {
      block_K = (K + num_blocks - 1) / num_blocks;
      for (k = 0; k < K; k += block_K) {
        block_K = block_K < K - k ? block_K : (K-k);
        switch(Order){
          case 'r':
          case 'R':
            switch(TransA){
              case 'n':
              case 'N':
                tmpA = A + k;
                break;
              default:
                tmpA = A + k * lda;
                break;
            }
            switch(TransB){
              case 'n':
              case 'N':
                tmpB = B + k * ldb;
                break;
              default:
                tmpB = B + k;
                break;
            }
            break;
          default:
            switch(TransA){
              case 'n':
              case 'N':
                tmpA = A + k * lda;
                break;
              default:
                tmpA = A + k;
                break;
            }
            switch(TransB){
              case 'n':
              case 'N':
                tmpB = B + k;
                break;
              default:
                tmpB = B + k * ldb;
                break;
            }
            break;
        }
        binnedBLAS_sbsgemm(fold, Order, TransA, TransB, M, N, block_K, alpha, tmpA, lda, tmpB, ldb, Ires, ldc);
      }
      for(i = 0; i < M; i++){
        for(j = 0; j < N; j++){
          switch(Order){
            case 'r':
            case 'R':
              res[i * ldc + j] = binned_ssbconv(fold, Ires + (i * ldc + j) * binned_sbnum(fold));
              break;
            default:
              res[j * ldc + i] = binned_ssbconv(fold, Ires + (j * ldc + i) * binned_sbnum(fold));
              break;
          }
        }
      }
    }
    for(i = 0; i < M; i++){
      for(j = 0; j < N; j++){
        switch(Order){
          case 'r':
          case 'R':
            if(res[i * ldc + j] != ref[i * ldc + j]){
              printf("reproBLAS_rsgemm(A, X, Y)[num_blocks=%d] = %g != %g\n", num_blocks, res[i * ldc + j], ref[i * ldc + j]);
              return 1;
            }
            break;
          default:
            if(res[j * ldc + i] != ref[j * ldc + i]){
              printf("reproBLAS_rsgemm(A, X, Y)[num_blocks=%d] = %g != %g\n", num_blocks, res[j * ldc + i], ref[j * ldc + i]);
              return 1;
            }
            break;
        }
      }
    }
    num_blocks *= 2;
  }
  free(res);
  free(Ires);
  return 0;
}

int matmat_fill_show_help(void){
  corroborate_rsgemm_options_initialize();

  opt_show_option(fold);
  opt_show_option(max_blocks);
  opt_show_option(shuffles);
  return 0;
}

const char* matmat_fill_name(int argc, char** argv){
  static char name_buffer[MAX_LINE];

  corroborate_rsgemm_options_initialize();

  opt_eval_option(argc, argv, &fold);
  opt_eval_option(argc, argv, &max_blocks);

  snprintf(name_buffer, MAX_LINE * sizeof(char), "Corroborate rsgemm fold=%d", fold._int.value);
  return name_buffer;
}

int matmat_fill_test(int argc, char** argv, char Order, char TransA, char TransB, int M, int N, int K, double RealAlpha, double ImagAlpha, int FillA, double RealScaleA, double ImagScaleA, int lda, int FillB, double RealScaleB, double ImagScaleB, int ldb, double RealBeta, double ImagBeta, int FillC, double RealScaleC, double ImagScaleC, int ldc){
  (void)ImagAlpha;
  (void)ImagBeta;
  int rc = 0;
  int i;
  int j;

  corroborate_rsgemm_options_initialize();

  opt_eval_option(argc, argv, &fold);
  opt_eval_option(argc, argv, &max_blocks);
  opt_eval_option(argc, argv, &shuffles);

  util_random_seed();
  char NTransA;
  int opAM;
  int opAK;
  int opBK;
  int opBN;

  switch(TransA){
    case 'n':
    case 'N':
      opAM = M;
      opAK = K;
      NTransA = 't';
      break;
    default:
      opAM = K;
      opAK = M;
      NTransA = 'n';
      break;
  }

  switch(TransB){
    case 'n':
    case 'N':
      opBK = K;
      opBN = N;
      break;
    default:
      opBK = N;
      opBN = K;
      break;
  }

  float *A  = util_smat_alloc(Order, opAM, opAK, lda);
  float *B  = util_smat_alloc(Order, opBK, opBN, ldb);
  float *C  = util_smat_alloc(Order, M, N, ldc);
  int CNM;
  switch(Order){
    case 'r':
    case 'R':
      CNM = M * ldc;
      break;
    default:
      CNM = ldc * N;
      break;
  }
  float_binned *CI = malloc(CNM * binned_sbsize(fold._int.value));

  int *P;

  util_smat_fill(Order, NTransA, opAM, opAK, A, lda, FillA, RealScaleA, ImagScaleA);
  util_smat_fill(Order, TransB, opBK, opBN, B, ldb, FillB, RealScaleB, ImagScaleB);
  util_smat_fill(Order, 'n', M, N, C, ldc, FillC, RealScaleC, ImagScaleC);
  for(i = 0; i < M; i++){
    for(j = 0; j < N; j++){
      switch(Order){
        case 'r':
        case 'R':
          binned_sbsconv(fold._int.value, C[i * ldc + j] * RealBeta, CI + (i * ldc + j) * binned_sbnum(fold._int.value));
          break;
        default:
          binned_sbsconv(fold._int.value, C[j * ldc + i] * RealBeta, CI + (j * ldc + i) * binned_sbnum(fold._int.value));
          break;
      }
    }
  }
  float *ref  = (float*)malloc(CNM * sizeof(float));

  //compute with unpermuted data
  memcpy(ref, C, CNM * sizeof(float));

  wrap_ref_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, ref, ldc);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  P = util_identity_permutation(K);
  util_smat_row_reverse(Order, NTransA, opAM, opAK, A, lda, P, 1);
  util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
  free(P);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  P = util_identity_permutation(K);
  util_smat_row_sort(Order, NTransA, opAM, opAK, A, lda, P, 1, util_Increasing, 0);
  util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
  free(P);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  P = util_identity_permutation(K);
  util_smat_row_sort(Order, NTransA, opAM, opAK, A, lda, P, 1, util_Decreasing, 0);
  util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
  free(P);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  P = util_identity_permutation(K);
  util_smat_row_sort(Order, NTransA, opAM, opAK, A, lda, P, 1, util_Increasing_Magnitude, 0);
  util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
  free(P);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  P = util_identity_permutation(K);
  util_smat_row_sort(Order, NTransA, opAM, opAK, A, lda, P, 1, util_Decreasing_Magnitude, 0);
  util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
  free(P);

  rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
  if(rc != 0){
    return rc;
  }

  for(i = 0; i < shuffles._int.value; i++){
    P = util_identity_permutation(K);
    util_smat_row_shuffle(Order, NTransA, opAM, opAK, A, lda, P, 1);
    util_smat_row_permute(Order, TransB, opBK, opBN, B, ldb, P, 1, NULL, 1);
    free(P);

    rc = corroborate_rsgemm(fold._int.value, Order, TransA, TransB, M, N, K, RealAlpha, A, lda, B, ldb, RealBeta, C, CI, ldc, ref, max_blocks._int.value);
    if(rc != 0){
      return rc;
    }
  }

  free(A);
  free(B);
  free(C);
  free(CI);
  free(ref);

  return rc;
}
