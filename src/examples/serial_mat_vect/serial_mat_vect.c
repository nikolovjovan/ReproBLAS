#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
#include <string.h>

static struct timeval start;
static struct timeval end;

void tic(void){
  gettimeofday( &start, NULL );
}

double toc(void){
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//helper to sum the vector
double sum_vector(double* vector, int n) {
  double sum;
  for (int i = 0; i < n; i++){
    sum = sum + vector[i];
  }
  return sum;
}


//function to shuffle the rows
void randomize(double** mat, int n) {
  // srand(time(NULL));
  for (int i = n-1; i>0; i--) {
    int j = rand() % (i+1);
    double* temp = mat[i];
    mat[i] = mat[j];
    mat[j] = temp;
  }
}

//helper to sum the array


//simple matrix-vector multiply
double vectors_dot_prod(const double *x, const double *y, int n){
    double res = 0.0;
    int i;
    for (i = 0; i < n; i++){
        res += x[i] * y[i];
    }
    return res;
}

void matrix_vector_mult(double **mat, const double *vec, double *result, int rows, int cols){
// in matrix form: result = mat * vec;
    int i;
    for (i = 0; i < rows; i++){
        result[i] = vectors_dot_prod(mat[i], vec, cols);
    }
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

int main(int argc, char** argv){
  double sum;
  double sum_shuffled;
  //srand();
  int n = 10000;
  double **matrix = malloc(n * sizeof(double));
  for (int i = 0; i < n; i++){
    matrix[i] = malloc(n * sizeof(double));
  }

  double *vector = malloc(n * sizeof(double));  
  double *result = malloc(n * sizeof(double));

  //printf("%s\n", "Vector is:");
  for (int i = 0; i < n; i++) {
    //vector[i] = rand();
    vector[i] = i + 1;
    //printf("%0.2f\n", vector[i]);
  }

  //printf("%s\n", "Original Matrix is:");
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      // matrix[i][j] = rand();
      matrix[i][j] = rand();
      //printf("%0.2f     ", matrix[i][j]);
    }
    //printf("\n");
  }

  matrix_vector_mult(matrix, vector, result, n, n);

  // printf("%s\n", "Result vector is:");
  // for (int i = 0; i < n; i++) {
  //   printf("%0.2f\n", result[i]);
  // }

  printf("%s\n", "Original Sum is:");
  //sum = reproBLAS_dsum(n, result, 1);
  sum = sum_vector(result,n);
  printf("%g\n", fabs(sum));

  //printf("%s\n", "Shuffled Matrix is:");
  randomize(matrix, n);
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%0.2f     ", matrix[i][j]);
  //   }
  //   printf("\n");
  // }  

  matrix_vector_mult(matrix, vector, result, n, n);

  // printf("%s\n", "Shuffled result vector is:");
  // for (int i = 0; i < n; i++) {
  //   printf("%0.2f\n", result[i]);
  // }

  printf("%s\n", "Shuffled Sum is:");
  //sum_shuffled = reproBLAS_dsum(n, result, 1);
  sum_shuffled = sum_vector(result,n);
  printf("%g\n", fabs(sum_shuffled));

  printf("%s : %g\n", "Difference is", fabs(sum - sum_shuffled));

  for (int i = 0; i < n; i++){
    free(matrix[i]);
  } 
  free(matrix);
  free(vector);
  free(result);
}
