/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

#include "../../../../config.h"
#include <binned.h>
#include <reproBLAS.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float **pts, /* [npts][nfeatures] */
                       int npts)
{
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}

/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
}

float **kmeans_clustering_seq(bool reproducible, bool manual,
                              float **feature, /* in: [npoints][nfeatures] */
                              int nfeatures, int npoints, int nclusters, float threshold,
                              int *membership) /* out: [npoints] */
{
    int i, j, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers; /* [nclusters][nfeatures] */
    float **clusters;    /* out: [nclusters][nfeatures] */
    float delta;

    float_binned ***new_centers_binned; /* [nclusters][nfeatures] */
    float *new_features;   // array of feature f of points nearest to current cluster
                           // used to sum all features of points belonging to a cluster at once

    /* allocate space for returning variable clusters[] */
    clusters = (float **) malloc(nclusters * sizeof(float *));
    clusters[0] = (float *) malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int) rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *) calloc(nclusters, sizeof(int));

    if (reproducible && manual) {
        new_centers_binned = (float_binned ***) malloc(nclusters * sizeof(float_binned **));
        new_centers_binned[0] = (float_binned **) malloc(nclusters * nfeatures * sizeof(float_binned *));
        for (i = 1; i < nclusters; i++)
            new_centers_binned[i] = new_centers_binned[i - 1] + nfeatures;
        for (i = 0; i < nclusters; ++i)
            for (j = 0; j < nfeatures; ++j) {
                new_centers_binned[i][j] = binned_sballoc(SIDEFAULTFOLD);
                binned_sbsetzero(SIDEFAULTFOLD, new_centers_binned[i][j]);
            }
    } else {
        new_centers = (float **) malloc(nclusters * sizeof(float *));
        new_centers[0] = (float *) calloc(nclusters * nfeatures, sizeof(float));
        for (i = 1; i < nclusters; i++)
            new_centers[i] = new_centers[i - 1] + nfeatures;

		if (reproducible) {
            /* allocate space for temporary feature array */
            new_features = (float *) malloc(npoints * sizeof(float));
        }
    }

    do {
        delta = 0.0;

        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;
            new_centers_len[index]++;

			if (!reproducible || manual) {
		        /* update new cluster centers : sum of objects located within */
	            for (j = 0; j < nfeatures; j++)
	                if (reproducible)
	                    binned_sbsadd(SIDEFAULTFOLD, feature[i][j], new_centers_binned[index][j]);
	                else
	                    new_centers[index][j] += feature[i][j];
        	}
        }

        if (reproducible && !manual) {
            for (index = 0; index < nclusters; index++) {
                /* update new cluster centers : sum of objects located within */
                for (j = 0; j < nfeatures; j++) {
                    n = 0; // number of features in current new_features array (goes to new_centers_len[index])
                    for (i = 0; i < npoints; i++) {
                        if (membership[i] == index) { // point i belongs to cluster index
                            new_features[n++] = feature[i][j];
                        }
                    }
                    new_centers[index][j] = reproBLAS_ssum(new_centers_len[index], new_features, 1);
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (reproducible && manual) {
                    if (new_centers_len[i] > 0)
                        clusters[i][j] = binned_ssbconv(SIDEFAULTFOLD, new_centers_binned[i][j]) / new_centers_len[i];
                    binned_sbsetzero(SIDEFAULTFOLD, new_centers_binned[i][j]); /* set back to 0 */
                } else {
                    if (new_centers_len[i] > 0)
                        clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                    new_centers[i][j] = 0.0; /* set back to 0 */
                }
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }
    } while (delta > threshold);

    if (reproducible && manual) {
        for (i = 0; i < nclusters; ++i)
            for (j = 0; j < nfeatures; ++j)
                free(new_centers_binned[i][j]);
        free(new_centers_binned[0]);
        free(new_centers_binned);
    } else {
        if (reproducible) {
            free(new_features);
        }

        free(new_centers[0]);
        free(new_centers);
    }

    free(new_centers_len);

    return clusters;
}

float **kmeans_clustering_omp(bool reproducible, bool manual,
                              float **feature, /* in: [npoints][nfeatures] */
                              int nfeatures, int npoints, int nclusters, float threshold,
                              int *membership) /* out: [npoints] */
{
    int i, j, k, l, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers; /* [nclusters][nfeatures] */
    float **clusters;    /* out: [nclusters][nfeatures] */
    float delta;

    int nthreads = omp_get_max_threads();
    int **partial_new_centers_len;
    float ***partial_new_centers;

    float_binned ***new_centers_binned; /* [nclusters][nfeatures] */
    float_binned ****partial_new_centers_binned;
    float **new_features;  // array of feature f of points nearest to current cluster
                           // used to sum all features of points belonging to a cluster at once

    /* allocate space for returning variable clusters[] */
    clusters = (float **) malloc(nclusters * sizeof(float *));
    clusters[0] = (float *) malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int) rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *) calloc(nclusters, sizeof(int));

    if (reproducible && manual) {
        new_centers_binned = (float_binned ***) malloc(nclusters * sizeof(float_binned **));
        new_centers_binned[0] = (float_binned **) malloc(nclusters * nfeatures * sizeof(float_binned *));
        for (i = 1; i < nclusters; i++)
            new_centers_binned[i] = new_centers_binned[i - 1] + nfeatures;
        for (i = 0; i < nclusters; ++i)
            for (j = 0; j < nfeatures; ++j) {
                new_centers_binned[i][j] = binned_sballoc(SIDEFAULTFOLD);
                binned_sbsetzero(SIDEFAULTFOLD, new_centers_binned[i][j]);
            }
    } else {
        new_centers = (float **) malloc(nclusters * sizeof(float *));
        new_centers[0] = (float *) calloc(nclusters * nfeatures, sizeof(float));
        for (i = 1; i < nclusters; i++)
            new_centers[i] = new_centers[i - 1] + nfeatures;
    }

    partial_new_centers_len = (int **) malloc(nthreads * sizeof(int *));
    partial_new_centers_len[0] = (int *) calloc(nthreads * nclusters, sizeof(int));
    for (i = 1; i < nthreads; ++i)
        partial_new_centers_len[i] = partial_new_centers_len[i - 1] + nclusters;

    if (reproducible && manual) {
        partial_new_centers_binned = (float_binned ****) malloc(nthreads * sizeof(float_binned ***));
        partial_new_centers_binned[0] = (float_binned ***) malloc(nthreads * nclusters * sizeof(float_binned **));
        for (i = 1; i < nthreads; i++)
            partial_new_centers_binned[i] = partial_new_centers_binned[i - 1] + nclusters;

        for (i = 0; i < nthreads; ++i)
            for (j = 0; j < nclusters; ++j) {
                partial_new_centers_binned[i][j] = (float_binned **) malloc(nfeatures * sizeof(float_binned *));
                for (k = 0; k < nfeatures; ++k) {
                    partial_new_centers_binned[i][j][k] = binned_sballoc(SIDEFAULTFOLD);
                    binned_sbsetzero(SIDEFAULTFOLD, partial_new_centers_binned[i][j][k]);
                }
            }
    } else {
        partial_new_centers = (float ***) malloc(nthreads * sizeof(float **));
        partial_new_centers[0] = (float **) malloc(nthreads * nclusters * sizeof(float *));
        for (i = 1; i < nthreads; ++i)
            partial_new_centers[i] = partial_new_centers[i - 1] + nclusters;

        for (i = 0; i < nthreads; ++i)
            for (j = 0; j < nclusters; ++j)
                partial_new_centers[i][j] = (float *) calloc(nfeatures, sizeof(float));

        if (reproducible) {
            /* allocate space for temporary feature array */
            new_features = (float **) malloc(nthreads * sizeof(float *));
            new_features[0] = (float *) malloc(nthreads * npoints * sizeof(float));
            for (i = 1; i < nthreads; ++i)
                new_features[i] = new_features[i - 1] + npoints;
        }
    }

    do {
        delta = 0.0;

#pragma omp parallel \
        private(i, j, k, l, n, index) \
        firstprivate(npoints, nclusters, nfeatures) \
        shared(feature, clusters, membership, partial_new_centers, partial_new_centers_len, new_features)
{
        int tid = omp_get_thread_num();

#pragma omp for \
        schedule(static) \
        reduction(+:delta)

        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;
            partial_new_centers_len[tid][index]++;

            if (!reproducible || manual) {
                /* update new cluster centers : sum of objects located within */
                for (j = 0; j < nfeatures; j++)
                    if (reproducible)
                        binned_sbsadd(SIDEFAULTFOLD, feature[i][j], partial_new_centers_binned[tid][index][j]);
                    else
                        partial_new_centers[tid][index][j] += feature[i][j];
            }
        }

        if (reproducible && !manual) {

#pragma omp for \
        schedule(static)

            /* reduce partial_new_centers_len */
            for (index = 0; index < nclusters; index++) {
                for (i = 0; i < nthreads; i++) {
                    new_centers_len[index] += partial_new_centers_len[i][index];
                    partial_new_centers_len[i][index] = 0;
                }
            }

            /* improve paralelization below since thread number > nclusters in the provided example */
            n = nclusters * nfeatures;

#pragma omp for \
        schedule(static)

            for (k = 0; k < n; k++) {
                index = k / nfeatures;
                j = k % nfeatures;
                l = 0; // number of points/features in current new_features array
                for (i = 0; i < npoints && l < new_centers_len[index]; i++) {
                    if (membership[i] == index) {
                        new_features[tid][l++] = feature[i][j];
                    }
                }
                new_centers[index][j] = reproBLAS_ssum(new_centers_len[index], new_features[tid], 1);
            }
        }
} /* end of #pragma omp parallel */

        if (!reproducible || manual) {
            /* let the main thread perform the array reduction */
            for (index = 0; index < nclusters; index++) {
                for (i = 0; i < nthreads; i++) {
                    new_centers_len[index] += partial_new_centers_len[i][index];
                    partial_new_centers_len[i][index] = 0.0;
                    if (reproducible)
                        for (j = 0; j < nfeatures; j++) {
                            binned_sbsbadd(SIDEFAULTFOLD, partial_new_centers_binned[i][index][j], new_centers_binned[index][j]);
                            binned_sbsetzero(SIDEFAULTFOLD, partial_new_centers_binned[i][index][j]);
                        }
                    else
                        for (j = 0; j < nfeatures; j++) {
	                        new_centers[index][j] += partial_new_centers[i][index][j];
	                        partial_new_centers[i][index][j] = 0.0;
                        }
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (index = 0; index < nclusters; index++) {
            for (j = 0; j < nfeatures; j++) {
                if (reproducible && manual) {
                    if (new_centers_len[index] > 0)
                        clusters[index][j] = binned_ssbconv(SIDEFAULTFOLD, new_centers_binned[index][j]) / new_centers_len[index];
                    binned_sbsetzero(SIDEFAULTFOLD, new_centers_binned[index][j]); /* set back to 0 */
                } else {
                    if (new_centers_len[index] > 0)
                        clusters[index][j] = new_centers[index][j] / new_centers_len[index];
                    new_centers[index][j] = 0.0; /* set back to 0 */
                }
            }
            new_centers_len[index] = 0; /* set back to 0 */
        }
    } while (delta > threshold && loop++ < 500);

    if (reproducible && manual) {
        for (i = 0; i < nthreads; ++i)
            for (j = 0; j < nclusters; ++j) {
                for (k = 0; k < nfeatures; ++k)
                    free(partial_new_centers_binned[i][j][k]);
                free(partial_new_centers_binned[i][j]);
            }
        free(partial_new_centers_binned[0]);
        free(partial_new_centers_binned);
    } else {
        if (reproducible) {
            free(new_features[0]);
            free(new_features);
        }

        for (i = 0; i < nthreads; ++i)
            for (index = 0; index < nclusters; ++index)
                free(partial_new_centers[i][index]);
        free(partial_new_centers[0]);
        free(partial_new_centers);
    }

    free(partial_new_centers_len[0]);
    free(partial_new_centers_len);

    if (reproducible && manual) {
        for (i = 0; i < nclusters; ++i)
            for (j = 0; j < nfeatures; ++j) {
                free(new_centers_binned[i][j]);
            }
        free(new_centers_binned[0]);
        free(new_centers_binned);
    } else {
        free(new_centers[0]);
        free(new_centers);
    }

    free(new_centers_len);

    return clusters;
}