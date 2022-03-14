#include <stdlib.h>

#include "../common/common.h"
#include "binnedBLAS.h"

/*[[[cog
import cog
import generate
import dataTypes
from src.common import blockSize
from scripts import terminal

rounded_cache = 2**(int(terminal.get_cache()).bit_length() - 1)
xy_block = rounded_cache//dataTypes.DoubleComplex.byte_size
y_block = max(xy_block//256, 1)

cog.out(generate.generate(blockSize.BlockSize("zbzgemm", "Y_BLOCK", 1, y_block, y_block, ["bench_rzgemm_AvgTransA_AvgTransB_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
cog.out(generate.generate(blockSize.BlockSize("zbzgemm", "XY_BLOCK", y_block, xy_block, xy_block, ["bench_rzgemm_AvgTransA_AvgTransB_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
cog.out(generate.generate(blockSize.BlockSize("zbzgemm", "YT_BLOCK", 1, y_block, y_block, ["bench_rzgemm_AvgTransA_AvgTransB_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
cog.out(generate.generate(blockSize.BlockSize("zbzgemm", "XYT_BLOCK", y_block, xy_block, xy_block, ["bench_rzgemm_AvgTransA_AvgTransB_fold_{}".format(terminal.get_didefaultfold())]), cog.inFile, args, params, mode))
]]]*/
#define Y_BLOCK 32
#define XY_BLOCK 16384
#define YT_BLOCK 32
#define XYT_BLOCK 16384
//[[[end]]]
#define X_BLOCK (XY_BLOCK/Y_BLOCK)
#define XT_BLOCK (XYT_BLOCK/YT_BLOCK)

/**
 * @brief Add to binned complex double precision matrix C the matrix-matrix product of complex double precision matrices A and B
 *
 * Performs one of the matrix-matrix operations
 *
 *   C := alpha*op(A)*op(B) + C,
 *
 * where op(X) is one of
 *
 *   op(X) = X   or   op(X) = X**T   or   op(X) = X**H,
 *
 * alpha is a scalar, A and B are matrices with op(A) an M by K matrix and op(B) a K by N matrix, and C is an binned M by N matrix.
 *
 * @param fold the fold of the binned types
 * @param Order a character specifying the matrix ordering ('r' or 'R' for row-major, 'c' or 'C' for column major)
 * @param TransA a character specifying whether or not to transpose A before taking the matrix-matrix product ('n' or 'N' not to transpose, 't' or 'T' to transpose, 'c' or 'C' to conjugate transpose)
 * @param TransB a character specifying whether or not to transpose B before taking the matrix-matrix product ('n' or 'N' not to transpose, 't' or 'T' to transpose, 'c' or 'C' to conjugate transpose)
 * @param M number of rows of matrix op(A) and of the matrix C.
 * @param N number of columns of matrix op(B) and of the matrix C.
 * @param K number of columns of matrix op(A) and columns of the matrix op(B).
 * @param alpha scalar alpha
 * @param A complex double precision matrix of dimension (ma, lda) in row-major or (lda, na) in column-major. (ma, na) is (M, K) if A is not transposed and (K, M) otherwise.
 * @param lda the first dimension of A as declared in the calling program. lda must be at least na in row major or ma in column major.
 * @param B complex double precision matrix of dimension (mb, ldb) in row-major or (ldb, nb) in column-major. (mb, nb) is (K, N) if B is not transposed and (N, K) otherwise.
 * @param ldb the first dimension of B as declared in the calling program. ldb must be at least nb in row major or mb in column major.
 * @param C binned complex double precision matrix of dimension (M, ldc) in row-major or (ldc, N) in column-major.
 * @param ldc the first dimension of C as declared in the calling program. ldc must be at least N in row major or M in column major.
 *
 * @author Peter Ahrens
 * @date   18 Jan 2016
 */
void binnedBLAS_zbzgemm(const int fold, const char Order,
             const char TransA, const char TransB,
             const int M, const int N, const int K,
             const void *alpha, const void *A, const int lda,
             const void *B, const int ldb,
             double_complex_binned *C, const int ldc){
  int i;
  int ii;
  int k;
  int kk;
  int j;
  int jj;
  double *bufA;
  int ldbufa;
  double *bufB;

  //early returns
  if(M == 0 || N == 0 || K == 0 || (((double*)alpha)[0] == 0.0 && ((double*)alpha)[1] == 0.0)){
    return;
  }

  switch(Order){

    //row major
    case 'r':
    case 'R':
      switch(TransA){

        //row major A not transposed
        case 'n':
        case 'N':
          if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
            bufA = (double*)A;
            ldbufa = lda;
          }else{
            bufA = (double*)malloc(M * K * 2 * sizeof(double));
            for(i = 0; i < M; i++){
              for(k = 0; k < K; k++){
                bufA[2 * (i * K + k)] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[0] - ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                bufA[2 * (i * K + k) + 1] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[1] + ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[0];
              }
            }
            ldbufa = K;
          }
          switch(TransB){

            //row major A not transposed B not transposed
            case 'n':
            case 'N':
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += YT_BLOCK){
                  for(k = 0; k < K; k += XT_BLOCK){
                    for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                      }
                    }
                    for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii * ldbufa + k), 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              free(bufB);
              break;

            //row major A not transposed B transposed
            case 't':
            case 'T':
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += Y_BLOCK){
                  for(k = 0; k < K; k += X_BLOCK){
                    for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        binnedBLAS_zbzdotu(fold, MIN(X_BLOCK, K - k), bufA + 2 * (ii * ldbufa + k), 1, (double*)B + 2 * (jj * ldb + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              break;

            //row major A not transposed B conjugate transposed
            default:
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += Y_BLOCK){
                  for(k = 0; k < K; k += X_BLOCK){
                    for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        binnedBLAS_zbzdotc(fold, MIN(X_BLOCK, K - k), (double*)B + 2 * (jj * ldb + k), 1, bufA + 2 * (ii * ldbufa + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              break;

          }
          if(((double*)alpha)[0] != 1.0 || ((double*)alpha)[1] != 0.0){
            free(bufA);
          }
          break;

        //row major A transposed
        case 't':
        case 'T':
          switch(TransB){

            //row major A transposed B not transposed
            case 'n':
            case 'N':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              free(bufB);
              break;

            //row major A transposed B transposed
            case 't':
            case 'T':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              break;

            //row major A transposed B conjugate transposed
            default:
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)B) + 2 * (jj * ldb + k), 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)B) + 2 * (jj * ldb + k), 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              break;

          }
          break;

        //row major A conjugate transposed
        default:
          switch(TransB){

            //row major A conjugate transposed B not transposed
            case 'n':
            case 'N':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              free(bufB);
              break;

            //row major A conjugate transposed B transposed
            case 't':
            case 'T':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              break;

            //row major A conjugate transposed B conjugate transposed
            default:
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = -((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)B) + 2 * (jj * ldb + k), 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                        for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)B) + 2 * (jj * ldb + k), 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (ii * ldc + jj) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              break;
          }
          break;
      }
      break;

    //column major
    default:
      switch(TransA){

        //column major A not transposed
        case 'n':
        case 'N':
          switch(TransB){

            //column major A not transposed B not transposed
            case 'n':
            case 'N':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              break;

            //column major A not transposed B transposed
            case 't':
            case 'T':
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii - i) * XT_BLOCK, 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              free(bufB);
              break;

            //column major A not transposed B conjugate transposed
            default:
              bufA = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii) + 1];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufB + 2 * (jj - j) * XT_BLOCK, 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                for(i = 0; i < M; i += YT_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k))] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[0] - ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[1];
                          bufA[2 * ((ii - i) * XT_BLOCK + (kk - k)) + 1] = ((double*)A)[2 * (kk * lda + ii)] * ((double*)alpha)[1] + ((double*)A)[2 * (kk * lda + ii) + 1] * ((double*)alpha)[0];
                        }
                      }
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + YT_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufB + 2 * (jj - j) * XT_BLOCK, 1, bufA + 2 * (ii - i) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }
              free(bufA);
              free(bufB);
              break;
          }
          break;

        //column major A transposed
        case 't':
        case 'T':
          if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
            bufA = (double*)A;
            ldbufa = lda;
          }else{
            bufA = (double*)malloc(M * K * 2 * sizeof(double));
            for(i = 0; i < M; i++){
              for(k = 0; k < K; k++){
                bufA[2 * (i * K + k)] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[0] - ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                bufA[2 * (i * K + k) + 1] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[1] + ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
              }
            }
            ldbufa = K;
          }
          switch(TransB){

            //column major A transposed B not transposed
            case 'n':
            case 'N':
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += Y_BLOCK){
                  for(k = 0; k < K; k += X_BLOCK){
                    for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                      for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                        binnedBLAS_zbzdotu(fold, MIN(X_BLOCK, K - k), bufA + 2 * (ii * ldbufa + k), 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              break;

            //column major A transposed B transposed
            case 't':
            case 'T':
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += YT_BLOCK){
                  for(k = 0; k < K; k += XT_BLOCK){
                    for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                      }
                    }
                    for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                      for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                        binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii * ldbufa + k), 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              free(bufB);
              break;

            //column major A transposed B conjugate transposed
            default:
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              for(i = 0; i < M; i += Y_BLOCK){
                for(j = 0; j < N; j += YT_BLOCK){
                  for(k = 0; k < K; k += XT_BLOCK){
                    for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                        bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                      }
                    }
                    for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                      for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                        binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufB + 2 * (jj - j) * XT_BLOCK, 1, bufA + 2 * (ii * ldbufa + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                      }
                    }
                  }
                }
              }
              free(bufB);
              break;
          }
          if(((double*)alpha)[0] != 1.0 || ((double*)alpha)[1] != 0.0){
            free(bufA);
          }
          break;

        //column major A conjugate transposed
        default:
          switch(TransB){

            //column major A conjugate transposed B not transposed
            case 'n':
            case 'N':
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += X_BLOCK){
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(X_BLOCK, K - k), ((double*)A) + 2 * (ii * lda + k), 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                bufA = (double*)malloc(M * K * 2 * sizeof(double));
                for(i = 0; i < M; i++){
                  for(k = 0; k < K; k++){
                    bufA[2 * (i * K + k)] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[0] + ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                    bufA[2 * (i * K + k) + 1] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[1] - ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                  }
                }
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += Y_BLOCK){
                    for(k = 0; k < K; k += X_BLOCK){
                      for(jj = j; jj < N && jj < j + Y_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(X_BLOCK, K - k), bufA + 2 * (ii * K + k), 1, ((double*)B) + 2 * (jj * ldb + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
                free(bufA);
              }
              break;

            //column major A conjugate transposed B transposed
            case 't':
            case 'T':
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)A) + 2 * (ii * lda + k), 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                bufA = (double*)malloc(M * K * 2 * sizeof(double));
                for(i = 0; i < M; i++){
                  for(k = 0; k < K; k++){
                    bufA[2 * (i * K + k)] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[0] + ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                    bufA[2 * (i * K + k) + 1] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[1] - ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                  }
                }
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotu(fold, MIN(XT_BLOCK, K - k), bufA + 2 * (ii * K + k), 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
                free(bufA);
              }
              free(bufB);
              break;

            //column major A conjugate transposed B conjugate transposed
            default:
              bufB = (double*)malloc(XT_BLOCK * YT_BLOCK * 2 * sizeof(double));
              if(((double*)alpha)[0] == 1.0 && ((double*)alpha)[1] == 0.0){
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = -((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), ((double*)A) + 2 * (ii * lda + k), 1, bufB + 2 * (jj - j) * XT_BLOCK, 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
              }else{
                bufA = (double*)malloc(M * K * 2 * sizeof(double));
                for(i = 0; i < M; i++){
                  for(k = 0; k < K; k++){
                    bufA[2 * (i * K + k)] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[0] + ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                    bufA[2 * (i * K + k) + 1] = ((double*)A)[2 * (i * lda + k)] * ((double*)alpha)[1] - ((double*)A)[2 * (i * lda + k) + 1] * ((double*)alpha)[1];
                  }
                }
                for(i = 0; i < M; i += Y_BLOCK){
                  for(j = 0; j < N; j += YT_BLOCK){
                    for(k = 0; k < K; k += XT_BLOCK){
                      for(kk = k; kk < K && kk < k + XT_BLOCK; kk++){
                        for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k))] = ((double*)B)[2 * (kk * ldb + jj)];
                          bufB[2 * ((jj - j) * XT_BLOCK + (kk - k)) + 1] = ((double*)B)[2 * (kk * ldb + jj) + 1];
                        }
                      }
                      for(jj = j; jj < N && jj < j + YT_BLOCK; jj++){
                        for(ii = i; ii < M && ii < i + Y_BLOCK; ii++){
                          binnedBLAS_zbzdotc(fold, MIN(XT_BLOCK, K - k), bufB + 2 * (jj - j) * XT_BLOCK, 1, bufA + 2 * (ii * K + k), 1, C + (jj * ldc + ii) * binned_zbnum(fold));
                        }
                      }
                    }
                  }
                }
                free(bufA);
              }
              free(bufB);
              break;
          }
          break;
      }
      break;
  }
}