#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "Common/Common.h"
#include <immintrin.h>
#include <emmintrin.h>

/*[[[cog
import cog
import sys, os
from gen import generate
from gen import dataTypes
from gen import vectorizations
import asumI2
]]]*/
//[[[end]]]

#if defined( __AVX__ )
  void dasumI2(int n, double* v, int incv, int fold, double* sum){
    /*[[[cog
    cog.out(generate.generate(asumI2.ASumI2(dataTypes.Double, vectorizations.AVX), args, params))
    ]]]*/
    __m256d mask_ABS; AVX_ABS_MASKD(mask_ABS);
    __m256d mask_BLP; AVX_BLP_MASKD(mask_BLP);
    double tmp_cons[4] __attribute__((aligned(32)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        __m256d v_0, v_1, v_2, v_3;
        __m256d q_0, q_1;
        __m256d s_0_0, s_0_1;
        __m256d s_1_0, s_1_1;
        __m256d s_2_0, s_2_1;

        s_0_0 = s_0_1 = _mm256_broadcast_sd(sum);
        s_1_0 = s_1_1 = _mm256_broadcast_sd(sum + 1);
        s_2_0 = s_2_1 = _mm256_broadcast_sd(sum + 2);
        if(incv == 1){

          for(i = 0; i + 16 <= n; i += 16, v += 16){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_loadu_pd(v + 4), mask_ABS);
            v_2 = _mm256_and_pd(_mm256_loadu_pd(v + 8), mask_ABS);
            v_3 = _mm256_and_pd(_mm256_loadu_pd(v + 12), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_loadu_pd(v + 4), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            i += 8, v += 8;
          }
          if(i + 4 <= n){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            i += 4, v += 4;
          }
          if(i < n){
            v_0 = _mm256_and_pd(_mm256_set_pd(0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 16 <= n; i += 16, v += (incv * 16)){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_set_pd(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            v_2 = _mm256_and_pd(_mm256_set_pd(v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), mask_ABS);
            v_3 = _mm256_and_pd(_mm256_set_pd(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)]), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_2, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_2, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_3, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_2 = _mm256_add_pd(v_2, q_0);
            v_3 = _mm256_add_pd(v_3, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_2, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_set_pd(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            s_0_1 = _mm256_add_pd(s_0_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            q_1 = _mm256_sub_pd(q_1, s_0_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            s_1_1 = _mm256_add_pd(s_1_1, _mm256_or_pd(v_1, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            q_1 = _mm256_sub_pd(q_1, s_1_1);
            v_0 = _mm256_add_pd(v_0, q_0);
            v_1 = _mm256_add_pd(v_1, q_1);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            s_2_1 = _mm256_add_pd(s_2_1, _mm256_or_pd(v_1, mask_BLP));
            i += 8, v += (incv * 8);
          }
          if(i + 4 <= n){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i < n){
            v_0 = _mm256_and_pd(_mm256_set_pd(0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm256_add_pd(s_0_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_0_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm256_add_pd(s_1_0, _mm256_or_pd(v_0, mask_BLP));
            q_0 = _mm256_sub_pd(q_0, s_1_0);
            v_0 = _mm256_add_pd(v_0, q_0);
            s_2_0 = _mm256_add_pd(s_2_0, _mm256_or_pd(v_0, mask_BLP));
          }
        }
        s_0_0 = _mm256_sub_pd(s_0_0, _mm256_set_pd(sum[0], sum[0], sum[0], 0));
        q_0 = _mm256_broadcast_sd(sum);
        s_0_0 = _mm256_add_pd(s_0_0, _mm256_sub_pd(s_0_1, q_0));
        _mm256_store_pd(tmp_cons, s_0_0);
        sum[0] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        s_1_0 = _mm256_sub_pd(s_1_0, _mm256_set_pd(sum[1], sum[1], sum[1], 0));
        q_0 = _mm256_broadcast_sd(sum + 1);
        s_1_0 = _mm256_add_pd(s_1_0, _mm256_sub_pd(s_1_1, q_0));
        _mm256_store_pd(tmp_cons, s_1_0);
        sum[1] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        s_2_0 = _mm256_sub_pd(s_2_0, _mm256_set_pd(sum[2], sum[2], sum[2], 0));
        q_0 = _mm256_broadcast_sd(sum + 2);
        s_2_0 = _mm256_add_pd(s_2_0, _mm256_sub_pd(s_2_1, q_0));
        _mm256_store_pd(tmp_cons, s_2_0);
        sum[2] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        __m256d v_0, v_1, v_2, v_3;
        __m256d q_0, q_1;
        __m256d s_0, s_1;
        __m256d s_buffer[(MAX_FOLD * 2)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm256_broadcast_sd(sum + j);
        }
        if(incv == 1){

          for(i = 0; i + 16 <= n; i += 16, v += 16){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_loadu_pd(v + 4), mask_ABS);
            v_2 = _mm256_and_pd(_mm256_loadu_pd(v + 8), mask_ABS);
            v_3 = _mm256_and_pd(_mm256_loadu_pd(v + 12), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_2, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_3, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_2 = _mm256_add_pd(v_2, q_0);
              v_3 = _mm256_add_pd(v_3, q_1);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_2, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_loadu_pd(v + 4), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(v_1, mask_BLP));
            i += 8, v += 8;
          }
          if(i + 4 <= n){
            v_0 = _mm256_and_pd(_mm256_loadu_pd(v), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
            i += 4, v += 4;
          }
          if(i < n){
            v_0 = _mm256_and_pd(_mm256_set_pd(0, (n - i)>2?v[2]:0, (n - i)>1?v[1]:0, v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 16 <= n; i += 16, v += (incv * 16)){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_set_pd(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            v_2 = _mm256_and_pd(_mm256_set_pd(v[(incv * 11)], v[(incv * 10)], v[(incv * 9)], v[(incv * 8)]), mask_ABS);
            v_3 = _mm256_and_pd(_mm256_set_pd(v[(incv * 15)], v[(incv * 14)], v[(incv * 13)], v[(incv * 12)]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_2, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_3, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_2 = _mm256_add_pd(v_2, q_0);
              v_3 = _mm256_add_pd(v_3, q_1);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_2, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(v_3, mask_BLP));
          }
          if(i + 8 <= n){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            v_1 = _mm256_and_pd(_mm256_set_pd(v[(incv * 7)], v[(incv * 6)], v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              q_1 = _mm256_add_pd(s_1, _mm256_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm256_sub_pd(s_0, q_0);
              q_1 = _mm256_sub_pd(s_1, q_1);
              v_0 = _mm256_add_pd(v_0, q_0);
              v_1 = _mm256_add_pd(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm256_add_pd(s_buffer[((j * 2) + 1)], _mm256_or_pd(v_1, mask_BLP));
            i += 8, v += (incv * 8);
          }
          if(i + 4 <= n){
            v_0 = _mm256_and_pd(_mm256_set_pd(v[(incv * 3)], v[(incv * 2)], v[incv], v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i < n){
            v_0 = _mm256_and_pd(_mm256_set_pd(0, (n - i)>2?v[(incv * 2)]:0, (n - i)>1?v[incv]:0, v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm256_add_pd(s_0, _mm256_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm256_sub_pd(s_0, q_0);
              v_0 = _mm256_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_or_pd(v_0, mask_BLP));
          }
        }
        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = _mm256_sub_pd(s_buffer[(j * 2)], _mm256_set_pd(sum[j], sum[j], sum[j], 0));
          q_0 = _mm256_broadcast_sd(sum + j);
          s_buffer[(j * 2)] = _mm256_add_pd(s_buffer[(j * 2)], _mm256_sub_pd(s_buffer[((j * 2) + 1)], q_0));
          _mm256_store_pd(tmp_cons, s_buffer[(j * 2)]);
          sum[j] = tmp_cons[0] + tmp_cons[1] + tmp_cons[2] + tmp_cons[3];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
    //[[[end]]]
  }
#elif defined( __SSE2__ )
  void dasumI2(int n, double* v, int incv, int fold, double* sum){
    /*[[[cog
    cog.out(generate.generate(asumI2.ASumI2(dataTypes.Double, vectorizations.SSE), args, params))
    ]]]*/
    __m128d mask_ABS; SSE_ABS_MASKD(mask_ABS);
    __m128d mask_BLP; SSE_BLP_MASKD(mask_BLP);
    double tmp_cons[2] __attribute__((aligned(16)));
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        __m128d v_0, v_1, v_2, v_3;
        __m128d q_0, q_1;
        __m128d s_0_0, s_0_1;
        __m128d s_1_0, s_1_1;
        __m128d s_2_0, s_2_1;

        s_0_0 = s_0_1 = _mm_load1_pd(sum);
        s_1_0 = s_1_1 = _mm_load1_pd(sum + 1);
        s_2_0 = s_2_1 = _mm_load1_pd(sum + 2);
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v += 8){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            v_1 = _mm_and_pd(_mm_loadu_pd(v + 2), mask_ABS);
            v_2 = _mm_and_pd(_mm_loadu_pd(v + 4), mask_ABS);
            v_3 = _mm_and_pd(_mm_loadu_pd(v + 6), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_2, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_2 = _mm_add_pd(v_2, q_0);
            v_3 = _mm_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_2, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_2 = _mm_add_pd(v_2, q_0);
            v_3 = _mm_add_pd(v_3, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_2, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            v_1 = _mm_and_pd(_mm_loadu_pd(v + 2), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            i += 4, v += 4;
          }
          if(i + 2 <= n){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            i += 2, v += 2;
          }
          if(i < n){
            v_0 = _mm_and_pd(_mm_set_pd(0, v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v += (incv * 8)){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            v_1 = _mm_and_pd(_mm_set_pd(v[(incv * 3)], v[(incv * 2)]), mask_ABS);
            v_2 = _mm_and_pd(_mm_set_pd(v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            v_3 = _mm_and_pd(_mm_set_pd(v[(incv * 7)], v[(incv * 6)]), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_2, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_2 = _mm_add_pd(v_2, q_0);
            v_3 = _mm_add_pd(v_3, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_2, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_3, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_2 = _mm_add_pd(v_2, q_0);
            v_3 = _mm_add_pd(v_3, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_2, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            v_1 = _mm_and_pd(_mm_set_pd(v[(incv * 3)], v[(incv * 2)]), mask_ABS);
            q_0 = s_0_0;
            q_1 = s_0_1;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            s_0_1 = _mm_add_pd(s_0_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            q_1 = _mm_sub_pd(q_1, s_0_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            q_0 = s_1_0;
            q_1 = s_1_1;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            s_1_1 = _mm_add_pd(s_1_1, _mm_or_pd(v_1, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            q_1 = _mm_sub_pd(q_1, s_1_1);
            v_0 = _mm_add_pd(v_0, q_0);
            v_1 = _mm_add_pd(v_1, q_1);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            s_2_1 = _mm_add_pd(s_2_1, _mm_or_pd(v_1, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i + 2 <= n){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
            i += 2, v += (incv * 2);
          }
          if(i < n){
            v_0 = _mm_and_pd(_mm_set_pd(0, v[0]), mask_ABS);
            q_0 = s_0_0;
            s_0_0 = _mm_add_pd(s_0_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_0_0);
            v_0 = _mm_add_pd(v_0, q_0);
            q_0 = s_1_0;
            s_1_0 = _mm_add_pd(s_1_0, _mm_or_pd(v_0, mask_BLP));
            q_0 = _mm_sub_pd(q_0, s_1_0);
            v_0 = _mm_add_pd(v_0, q_0);
            s_2_0 = _mm_add_pd(s_2_0, _mm_or_pd(v_0, mask_BLP));
          }
        }
        s_0_0 = _mm_sub_pd(s_0_0, _mm_set_pd(sum[0], 0));
        q_0 = _mm_load1_pd(sum);
        s_0_0 = _mm_add_pd(s_0_0, _mm_sub_pd(s_0_1, q_0));
        _mm_store_pd(tmp_cons, s_0_0);
        sum[0] = tmp_cons[0] + tmp_cons[1];
        s_1_0 = _mm_sub_pd(s_1_0, _mm_set_pd(sum[1], 0));
        q_0 = _mm_load1_pd(sum + 1);
        s_1_0 = _mm_add_pd(s_1_0, _mm_sub_pd(s_1_1, q_0));
        _mm_store_pd(tmp_cons, s_1_0);
        sum[1] = tmp_cons[0] + tmp_cons[1];
        s_2_0 = _mm_sub_pd(s_2_0, _mm_set_pd(sum[2], 0));
        q_0 = _mm_load1_pd(sum + 2);
        s_2_0 = _mm_add_pd(s_2_0, _mm_sub_pd(s_2_1, q_0));
        _mm_store_pd(tmp_cons, s_2_0);
        sum[2] = tmp_cons[0] + tmp_cons[1];
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        __m128d v_0, v_1, v_2, v_3;
        __m128d q_0, q_1;
        __m128d s_0, s_1;
        __m128d s_buffer[(MAX_FOLD * 2)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = _mm_load1_pd(sum + j);
        }
        if(incv == 1){

          for(i = 0; i + 8 <= n; i += 8, v += 8){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            v_1 = _mm_and_pd(_mm_loadu_pd(v + 2), mask_ABS);
            v_2 = _mm_and_pd(_mm_loadu_pd(v + 4), mask_ABS);
            v_3 = _mm_and_pd(_mm_loadu_pd(v + 6), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_2, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_3, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_2 = _mm_add_pd(v_2, q_0);
              v_3 = _mm_add_pd(v_3, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            v_1 = _mm_and_pd(_mm_loadu_pd(v + 2), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(v_1, mask_BLP));
            i += 4, v += 4;
          }
          if(i + 2 <= n){
            v_0 = _mm_and_pd(_mm_loadu_pd(v), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
            i += 2, v += 2;
          }
          if(i < n){
            v_0 = _mm_and_pd(_mm_set_pd(0, v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
          }
        }else{

          for(i = 0; i + 8 <= n; i += 8, v += (incv * 8)){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            v_1 = _mm_and_pd(_mm_set_pd(v[(incv * 3)], v[(incv * 2)]), mask_ABS);
            v_2 = _mm_and_pd(_mm_set_pd(v[(incv * 5)], v[(incv * 4)]), mask_ABS);
            v_3 = _mm_and_pd(_mm_set_pd(v[(incv * 7)], v[(incv * 6)]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_2, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_3, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_2 = _mm_add_pd(v_2, q_0);
              v_3 = _mm_add_pd(v_3, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_2, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(v_3, mask_BLP));
          }
          if(i + 4 <= n){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            v_1 = _mm_and_pd(_mm_set_pd(v[(incv * 3)], v[(incv * 2)]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              q_1 = _mm_add_pd(s_1, _mm_or_pd(v_1, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = _mm_sub_pd(s_0, q_0);
              q_1 = _mm_sub_pd(s_1, q_1);
              v_0 = _mm_add_pd(v_0, q_0);
              v_1 = _mm_add_pd(v_1, q_1);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
            s_buffer[((j * 2) + 1)] = _mm_add_pd(s_buffer[((j * 2) + 1)], _mm_or_pd(v_1, mask_BLP));
            i += 4, v += (incv * 4);
          }
          if(i + 2 <= n){
            v_0 = _mm_and_pd(_mm_set_pd(v[incv], v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
            i += 2, v += (incv * 2);
          }
          if(i < n){
            v_0 = _mm_and_pd(_mm_set_pd(0, v[0]), mask_ABS);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              q_0 = _mm_add_pd(s_0, _mm_or_pd(v_0, mask_BLP));
              s_buffer[(j * 2)] = q_0;
              q_0 = _mm_sub_pd(s_0, q_0);
              v_0 = _mm_add_pd(v_0, q_0);
            }
            s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_or_pd(v_0, mask_BLP));
          }
        }
        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = _mm_sub_pd(s_buffer[(j * 2)], _mm_set_pd(sum[j], 0));
          q_0 = _mm_load1_pd(sum + j);
          s_buffer[(j * 2)] = _mm_add_pd(s_buffer[(j * 2)], _mm_sub_pd(s_buffer[((j * 2) + 1)], q_0));
          _mm_store_pd(tmp_cons, s_buffer[(j * 2)]);
          sum[j] = tmp_cons[0] + tmp_cons[1];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
    //[[[end]]]
  }
#else
  void dasumI2(int n, double* v, int incv, int fold, double* sum){
    /*[[[cog
    cog.out(generate.generate(asumI2.ASumI2(dataTypes.Double, vectorizations.SISD), args, params))
    ]]]*/
    long_double tmp_BLP;
    SET_DAZ_FLAG;
    switch(fold){
      case 3:{
        int i;

        double v_0, v_1, v_2, v_3;
        double q_0, q_1;
        double s_0_0, s_0_1;
        double s_1_0, s_1_1;
        double s_2_0, s_2_1;

        s_0_0 = s_0_1 = sum[0];
        s_1_0 = s_1_1 = sum[1];
        s_2_0 = s_2_1 = sum[2];
        if(incv == 1){

          for(i = 0; i + 4 <= n; i += 4, v += 4){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[1]);
            v_2 = fabs(v[2]);
            v_3 = fabs(v[3]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_2 = v_2 + q_0;
            v_3 = v_3 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_2 = v_2 + q_0;
            v_3 = v_3 + q_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
          }
          if(i + 2 <= n){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[1]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            i += 2, v += 2;
          }
          if(i + 1 <= n){
            v_0 = fabs(v[0]);
            q_0 = s_0_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            v_0 = v_0 + q_0;
            q_0 = s_1_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            v_0 = v_0 + q_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            i += 1, v += 1;
          }
        }else{

          for(i = 0; i + 4 <= n; i += 4, v += (incv * 4)){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[incv]);
            v_2 = fabs(v[(incv * 2)]);
            v_3 = fabs(v[(incv * 3)]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_2 = v_2 + q_0;
            v_3 = v_3 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_2 = v_2 + q_0;
            v_3 = v_3 + q_1;
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
          }
          if(i + 2 <= n){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[incv]);
            q_0 = s_0_0;
            q_1 = s_0_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_0_1 = s_0_1 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            q_1 = q_1 - s_0_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            q_0 = s_1_0;
            q_1 = s_1_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_1_1 = s_1_1 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            q_1 = q_1 - s_1_1;
            v_0 = v_0 + q_0;
            v_1 = v_1 + q_1;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_2_1 = s_2_1 + tmp_BLP.d;
            i += 2, v += (incv * 2);
          }
          if(i + 1 <= n){
            v_0 = fabs(v[0]);
            q_0 = s_0_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_0_0 = s_0_0 + tmp_BLP.d;
            q_0 = q_0 - s_0_0;
            v_0 = v_0 + q_0;
            q_0 = s_1_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_1_0 = s_1_0 + tmp_BLP.d;
            q_0 = q_0 - s_1_0;
            v_0 = v_0 + q_0;
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_2_0 = s_2_0 + tmp_BLP.d;
            i += 1, v += incv;
          }
        }
        q_0 = sum[0];
        s_0_0 = s_0_0 + (s_0_1 - q_0);
        sum[0] = s_0_0;
        q_0 = sum[1];
        s_1_0 = s_1_0 + (s_1_1 - q_0);
        sum[1] = s_1_0;
        q_0 = sum[2];
        s_2_0 = s_2_0 + (s_2_1 - q_0);
        sum[2] = s_2_0;
        RESET_DAZ_FLAG
        return;
      }
      default:{
        int i, j;

        double v_0, v_1, v_2, v_3;
        double q_0, q_1;
        double s_0, s_1;
        double s_buffer[(MAX_FOLD * 2)];

        for(j = 0; j < fold; j += 1){
          s_buffer[(j * 2)] = s_buffer[((j * 2) + 1)] = sum[j];
        }
        if(incv == 1){

          for(i = 0; i + 4 <= n; i += 4, v += 4){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[1]);
            v_2 = fabs(v[2]);
            v_3 = fabs(v[3]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_2;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_3;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_2 = v_2 + q_0;
              v_3 = v_3 + q_1;
            }
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.d;
          }
          if(i + 2 <= n){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[1]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.d;
            i += 2, v += 2;
          }
          if(i + 1 <= n){
            v_0 = fabs(v[0]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              q_0 = s_0 - q_0;
              v_0 = v_0 + q_0;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            i += 1, v += 1;
          }
        }else{

          for(i = 0; i + 4 <= n; i += 4, v += (incv * 4)){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[incv]);
            v_2 = fabs(v[(incv * 2)]);
            v_3 = fabs(v[(incv * 3)]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_2;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_3;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_2 = v_2 + q_0;
              v_3 = v_3 + q_1;
            }
            tmp_BLP.d = v_2;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            tmp_BLP.d = v_3;
            tmp_BLP.l |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.d;
          }
          if(i + 2 <= n){
            v_0 = fabs(v[0]);
            v_1 = fabs(v[incv]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              s_1 = s_buffer[((j * 2) + 1)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              tmp_BLP.d = v_1;
              tmp_BLP.l |= 1;
              q_1 = s_1 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              s_buffer[((j * 2) + 1)] = q_1;
              q_0 = s_0 - q_0;
              q_1 = s_1 - q_1;
              v_0 = v_0 + q_0;
              v_1 = v_1 + q_1;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            tmp_BLP.d = v_1;
            tmp_BLP.l |= 1;
            s_buffer[((j * 2) + 1)] = s_buffer[((j * 2) + 1)] + tmp_BLP.d;
            i += 2, v += (incv * 2);
          }
          if(i + 1 <= n){
            v_0 = fabs(v[0]);
            for(j = 0; j < fold - 1; j++){
              s_0 = s_buffer[(j * 2)];
              tmp_BLP.d = v_0;
              tmp_BLP.l |= 1;
              q_0 = s_0 + tmp_BLP.d;
              s_buffer[(j * 2)] = q_0;
              q_0 = s_0 - q_0;
              v_0 = v_0 + q_0;
            }
            tmp_BLP.d = v_0;
            tmp_BLP.l |= 1;
            s_buffer[(j * 2)] = s_buffer[(j * 2)] + tmp_BLP.d;
            i += 1, v += incv;
          }
        }
        for(j = 0; j < fold; j += 1){
          q_0 = sum[j];
          s_buffer[(j * 2)] = s_buffer[(j * 2)] + (s_buffer[((j * 2) + 1)] - q_0);
          sum[j] = s_buffer[(j * 2)];
        }
        RESET_DAZ_FLAG
        return;
      }
    }
    //[[[end]]]
  }
#endif
