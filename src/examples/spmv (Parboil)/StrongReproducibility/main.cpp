/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "parboil.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>

#include "file.h"
#include "convert_dataset.h"

#include <omp.h>
#include "../../../../config.h"
#include <binned.h>
#include <reproBLAS.h>

using namespace std;

constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr uint32_t DEFAULT_NUMBER_OF_RUNS = 50;

constexpr char* input_files[] = { "bcsstk32.mtx", "fidapm05.mtx", "jgl009.mtx" };

bool generate_vector (float *x_vector, int dim, uint32_t seed)
{
    if (nullptr == x_vector) {
        return false;
    }

    mt19937 gen(seed);
    uniform_real_distribution<float> float_dist(0.f, 1.f);

    for (int i = 0; i < dim; i++) {
        x_vector[i] = float_dist(gen);
    }

    return true;
}

bool diff(int dim, float *h_Ax_vector_1, float *h_Ax_vector_2)
{
    for (int i = 0; i < dim; i++)
        if (h_Ax_vector_1[i] != h_Ax_vector_2[i])
            return true;
    return false;
}

void spmv_seq (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector, bool manual = false)
{
    float sum = 0.0f;
    float_binned* sum_binned = manual ? binned_sballoc(SIDEFAULTFOLD) : nullptr;
    float* products = manual ? nullptr : (float*) malloc(dim * sizeof(float));

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
    for (int i = 0; i < dim; i++) {
        if (!reproducible) {
            sum = 0.0f;
        } else if (manual) {
            binned_sbsetzero(SIDEFAULTFOLD, sum_binned);
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                if (manual) {
                    binned_sbsadd(SIDEFAULTFOLD, d * t, sum_binned);
                } else {
                    products[k] = d * t;
                }
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            if (manual) {
                h_Ax_vector[h_perm[i]] = binned_ssbconv(SIDEFAULTFOLD, sum_binned);
            } else {
                h_Ax_vector[h_perm[i]] = reproBLAS_ssum(bound, products, 1);
            }
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }

    if (manual) {
        free(sum_binned);
    } else {
        free(products);
    }
}

void spmv_omp (bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
               float *h_x_vector, int *h_perm, float *h_Ax_vector, bool manual = false)
{
#pragma omp parallel
{
    float sum = 0.0f;
    float_binned* sum_binned = manual ? binned_sballoc(SIDEFAULTFOLD) : nullptr;
    float* products = manual ? nullptr : (float*) malloc(dim * sizeof(float));

    // Consider creating a random map by creating an array 0..dim - 1 and randomly shuffling it
    // for each execution. This should provide required randomness given the order of operations
    // is sequential at the moment.
    //
#pragma omp for
    for (int i = 0; i < dim; i++) {
        if (!reproducible) {
            sum = 0.0f;
        } else if (manual) {
            binned_sbsetzero(SIDEFAULTFOLD, sum_binned);
        }

        int bound = h_nzcnt[i];

        for (int k = 0; k < bound; k++) {
            int j = h_ptr[k] + i;
            int in = h_indices[j];

            float d = h_data[j];
            float t = h_x_vector[in];

            if (reproducible) {
                if (manual) {
                    binned_sbsadd(SIDEFAULTFOLD, d * t, sum_binned);
                } else {
                    products[k] = d * t;
                }
            } else {
                sum += d * t;
            }
        }

        if (reproducible) {
            if (manual) {
                h_Ax_vector[h_perm[i]] = binned_ssbconv(SIDEFAULTFOLD, sum_binned);
            } else {
                h_Ax_vector[h_perm[i]] = reproBLAS_ssum(bound, products, 1);
            }
        } else {
            h_Ax_vector[h_perm[i]] = sum;
        }
    }

    if (manual) {
        free(sum_binned);
    } else {
        free(products);
    }
}
}

void execute (uint32_t nruns, bool parallel, bool reproducible, int dim, int *h_nzcnt, int *h_ptr, int *h_indices, float *h_data,
              float *h_x_vector, int *h_perm, float *h_Ax_vector, double &time, bool manual = false)
{
    time = 0.0f;

    float *tmp_h_Ax_vector = new float[dim];

    for (int i = 0; i < nruns; ++i) {
        if (i == 0)
            time = omp_get_wtime();
        if (parallel)
            spmv_omp(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector, manual);
        else
            spmv_seq(reproducible, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, tmp_h_Ax_vector, manual);
        if (i == 0) {
            time = omp_get_wtime() - time;
            cout << fixed << setprecision(10) << (float) time * 1000.0 << '\t'; // ms
            memcpy (h_Ax_vector, tmp_h_Ax_vector, dim * sizeof (float));
        } else if (diff(dim, h_Ax_vector, tmp_h_Ax_vector)) {
            printf("%s (%sreproducible) implementation not reproducible after %d runs!\n",
                    parallel ? "Parallel" : "Sequential", reproducible ? "" : "non-", i);
            break;
        }
    }

    delete[] tmp_h_Ax_vector;
}

int main (int argc, char** argv)
{
    // Parameters declaration
    //
    int len;
    int depth;
    int dim;
    int pad=1;
    int nzcnt_len;

    // Host memory allocation
    // Matrix
    //
    float *h_data;
    int *h_indices;
    int *h_ptr;
    int *h_perm;
    int *h_nzcnt;

    // Vector
    //
    float *h_Ax_vector_seq_rep, *h_Ax_vector_omp_rep;
    float *h_x_vector;

    double time_seq_rep, time_omp_rep;

    const int exe_path_len = strrchr(argv[0], '/') - argv[0] + 1;
    char exe_path[256];
    strncpy(exe_path, argv[0], exe_path_len);
    exe_path[exe_path_len] = '\0';

    char input_file_path[256];

    cout << "unit: [ms]\n\n";

    cout << "reproducible results only\n\n";

    for (int i = 0; i < 3; ++i)
    {
        strncpy(input_file_path, exe_path, exe_path_len + 1);
        strcat(input_file_path, "data/");
        strcat(input_file_path, input_files[i]);

        cout << input_files[i] << "\n\n";

        int col_count;
        coo_to_jds(
            input_file_path,
            1, // row padding
            pad, // warp size
            1, // pack size
            0, // debug level [0:2]
            &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
            &col_count, &dim, &len, &nzcnt_len, &depth
        );

        h_x_vector = new float[dim];

        if (!generate_vector(h_x_vector, dim, DEFAULT_SEED)) {
            fprintf(stderr, "Failed to generate dense vector.\n");
            exit(-1);
        }

        // library

        h_Ax_vector_seq_rep = new float[dim];
        h_Ax_vector_omp_rep = new float[dim];

        cout << "\nseq\t";

        for (int run = 0; run < 3; ++run) execute (1, false, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq_rep, time_seq_rep);
        cout << '\n';
        
        for (int thread_count = 1; thread_count <= 128; thread_count <<= 1)
        {
            omp_set_dynamic(0);                 // Explicitly disable dynamic teams
            omp_set_num_threads(thread_count);  // Use  thread_count threads for all consecutive parallel regions

            #pragma omp parallel
            #pragma omp single
            {
                cout << omp_get_num_threads() << '\t';
            }

            for (int run = 0; run < 3; ++run) execute (1, true, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp_rep, time_omp_rep);
            cout << '\n';
        }

        cout << '\n';

        // manual

        cout << "\nmanual seq\t";

        for (int run = 0; run < 3; ++run) execute (1, false, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_seq_rep, time_seq_rep, true);
        cout << '\n';
        
        for (int thread_count = 1; thread_count <= 128; thread_count <<= 1)
        {
            omp_set_dynamic(0);                 // Explicitly disable dynamic teams
            omp_set_num_threads(thread_count);  // Use  thread_count threads for all consecutive parallel regions

            #pragma omp parallel
            #pragma omp single
            {
                cout << "manual " << omp_get_num_threads() << '\t';
            }

            for (int run = 0; run < 3; ++run) execute (1, true, true, dim, h_nzcnt, h_ptr, h_indices, h_data, h_x_vector, h_perm, h_Ax_vector_omp_rep, time_omp_rep, true);
            cout << '\n';
        }

        cout << '\n';

        delete[] h_data;
        delete[] h_indices;
        delete[] h_ptr;
        delete[] h_perm;
        delete[] h_nzcnt;
        delete[] h_Ax_vector_seq_rep;
        delete[] h_Ax_vector_omp_rep;
        delete[] h_x_vector;
    }

    return 0;
}