#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

static struct timeval start;
static struct timeval end;

void tic(void){
  gettimeofday( &start, NULL );
}

double toc(void){
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void doubledouble_plus_double(double* a, double b){
  double bv;
  double s1, s2, t1, t2;

  // Add two hi words
  s1 = a[0] + b;
  bv = s1 - a[0];
  s2 = ((b - bv) + (a[0] - (s1 - bv)));

  t1 = a[1] + s2;
  bv = t1 - a[1];
  t2 = ((s2 - bv) + (a[1] - (t1 - bv)));

  s2 = t1;

  // Renormalize (s1, s2)  to  (t1, s2)
  t1 = s1 + s2;
  t2 += s2 - (t1 - s1);

  // Renormalize (t1, t2)
  a[0] = t1 + t2;
  a[1] = t2 - (a[0] - t1);
}

double partitioned_reproblas_sum(double* toSum, int toSumSize, int partitions){
  
   int partitionSize = toSumSize/partitions;

   double_binned **repro_sum_bins = (double_binned**)malloc(partitions * binned_dbsize(3));
   double_binned *tmp_sum = (double_binned*)malloc(binned_dbsize(3));
   double_binned *repro_sum = (double_binned*)malloc(binned_dbsize(3));

   
   for (int i = 0; i < partitions; i++){
     repro_sum_bins[i] = (double_binned*)malloc(binned_dbsize(3)); 
   }
  
  for(int i = 0; i < partitions; i++){
      binned_dbsetzero(3, repro_sum_bins[i]);
      for (int j = i * partitionSize; j < (i + 1) * partitionSize; j++){
         binned_dbdconv(3, toSum[j], tmp_sum);
         binned_dbdbadd(3, tmp_sum, repro_sum_bins[i]);
      }
  }
  
  for(int i = 0; i < partitions; i++){
   binned_dbdbadd(3, repro_sum_bins[i], repro_sum);
  }  
  
   for (int i = 0; i < partitions; i++){
       free(repro_sum_bins[i]); 
   }
   double sum  =  binned_ddbconv(3, repro_sum);
   free(repro_sum_bins);
   free(tmp_sum);
   free(repro_sum);
   return sum;
  
}

int main(int argc, char** argv){
  
  int n = 1000000;
  double *x = malloc(n * sizeof(double));
  int n_bins = 200;
  int n_more_bins = 100;
  int bin_size;
  int more_bin_size;
  
  double *partial_bins = malloc(n_bins * sizeof(double));
  double *partial_more_bins = malloc(n_more_bins * sizeof(double));
 
  double sum_bins;
  double sum_more_bins;
  double elapsed_time;
  
  // Set x to be a sine wave
  for(int i = 0; i < n; i++){
    x[i] = sin(2 * M_PI * (i / (double)n - 0.5));
  }
  
  // Make a header
  printf("%15s : Time (s) : |Sum - Sum of Bins| = ?\n", "Sum Method");
  
  // First, we sum x using double precision
  tic();
  
  sum_bins = 0;
  sum_more_bins = 0;
  bin_size  = n/n_bins;
  more_bin_size  = n/n_more_bins;
  
  for(int i = 0; i < n_bins; i++){
	partial_bins[i] = 0;
  }
  for(int i = 0; i < n_more_bins; i++){
	partial_more_bins[i] = 0;
  }
  
  for(int i = 0; i < n_bins; i++){
	  for (int j = i * bin_size; j < (i + 1) * bin_size; j++)
      partial_bins[i] += x[j];
  }
  elapsed_time = toc();
  
  for(int i = 0; i < n_more_bins; i++){
	  for (int j = i * more_bin_size; j < (i + 1) * more_bin_size; j++)
		partial_more_bins[i] += x[j];
  }
  
  for (int i = 0; i < n_bins; i++){
	  sum_bins += partial_bins[i];
  }
  
  for (int i = 0; i < n_more_bins; i++){
	  sum_more_bins += partial_more_bins[i];
  }
  
  printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "double", elapsed_time, sum_bins, sum_more_bins, fabs(sum_bins - sum_more_bins));
  
  
  // Sum using ReproBlas
  
  tic();
  sum_bins = reproBLAS_dsum(n_bins, partial_bins, 1);
  elapsed_time = toc();

  tic();
  sum_more_bins = reproBLAS_dsum(n_more_bins, partial_more_bins, 1);
  elapsed_time = toc();

  //printf("%15s : %-8g : |%.17e - %.17e| = %g\n", "Reproblas sum", elapsed_time, sum_bins, sum_more_bins, fabs(sum_bins - sum_more_bins));
  
 // Summing Use ReproBlas Primitives 
 
 // Make a header
   //printf("Sum: %.17e \n" , partitioned_reproblas_sum(x, n, 1));
  printf("%15s : |Sum with a partition - Sum with different partition| = ?\n", " # partitions");
  double diffSum = partitioned_reproblas_sum(x, n, n_bins) - partitioned_reproblas_sum(x, n, n_more_bins);
  printf("%7i/%8i: |%.17e - %.17e| = %g\n", n_bins, n_more_bins, partitioned_reproblas_sum(x, n , n_bins), partitioned_reproblas_sum(x, n, n_more_bins), diffSum);
  
  free(x);
  
}
