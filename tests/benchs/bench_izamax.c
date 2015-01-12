#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/test_opt.h"
#include "../common/test_vec.h"
#include "../common/test_time.h"
#include "../common/test_perf.h"
#include "../common/blas_inc.h"

#include "vecvec_fill_bench_header.h"

#define FLOP_PER_N 2

int vecvec_fill_bench_desc(void){
  char *op_names[] = {"d_add", "d_cmp", "d_orb"};
  int op_counts[] = {1, 1, 2};
  perf_output_desc(3, op_names, op_counts);
  return 0;
}

int vecvec_fill_bench_show_help(void){
  return 0;
}

const char* vecvec_fill_bench_name(int argc, char** argv){
  static char name_buffer[MAX_LINE];
  snprintf(name_buffer, MAX_LINE * sizeof(char), "Benchmark [izamax]");
  return name_buffer;
}

int vecvec_fill_bench_test(int argc, char** argv, int N, int incx, int incy, int type, double scale, double cond, int trials){
  int rc = 0;
  double complex res;
  double complex *x = zvec_alloc(N, incx);

  vec_random_seed();

  //fill empty space with random data to check increment
  zvec_fill(N * incx, x, 1, vec_fill_RAND, 1.0, 1.0);

  //fill x
  zvec_fill(N, x, incx, type, scale, cond);

  time_tic();
  for(int i = 0; i < trials; i++){
    CALL_IZAMAX(res, N, x, incx);
  }
  time_toc();

  perf_output_perf(time_read(), N, trials);

  free(x);
  return rc;
}
