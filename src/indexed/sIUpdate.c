/*
 *  Created   13/10/25   H.D. Nguyen
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "indexed.h"
#include "../Common/Common.h"

#ifdef __SSE__
#	include <xmmintrin.h>
#endif

#define PREC 24

// CHANGE THE INDEX
// IN FACT, IT IS A POSSIBLE SHIFTING LEFT
void sIUpdate_work(int fold, int NB, float step,
	float* x, float* c, int ldx, float y) {
	int i;
	int d;

	if (y == 0)
		return;

	if (x[0] == 0) {
		sIBoundary_(fold, fabs(y), x, ldx);
		for (i = fold; i < 2 * fold; i++) x[i * ldx] = 0;
		return;
	}


	float M = ufpf(x[0]);
//	printf("x=%g, M = %g.", x[0], M);

	M = M / NB;

	// NO NEED TO UPDATE
	if (y < M)
		return;

	step = 1.0/step;
	
	d = 0;
	// EVALUATE SHIFTING DISTANCE
	while (y >= M && d < fold) { d++; M *= step; }

	// RIGHT-SHIFTING
	for (i = fold-1; i >= d; i--) {
		x[i * ldx] = x[(i - d) * ldx];
		c[i * ldx] = c[(i - d) * ldx];
	}

	sIBoundary_(d, fabs(y), x, ldx);
	for (i = 0; i < d; i++) c[i * ldx] = 0.0;
}

void sIUpdate1(int fold, float y, float* x,
	float* c, int ldx){
	if (y == 0 || isnan(y) || isinf(y))
		return;

	float M = ufpf(x[0]);
	int W = sIWidth();
	int NB     = 1 << (PREC - W);
	M = M / NB;
	float step = ldexp(0.5f, 1-W);

	sIUpdate_work(fold, NB, step, x, c, ldx, y);
}

void cIUpdates1(int K, float complex* X, float* C, int INC, float Y) {
	Y = fabs(Y);
	sIUpdate1(K,Y,(float*)X  , C  , 2*INC);
	sIUpdate1(K,Y,(float*)X+1, C+1, 2*INC);
}

void cIUpdate1(int K, float complex* X, float* C,int INC, float complex Y) {
	float* tmp = (float*)&Y;
	sIUpdate1(K, fabs(tmp[0]), (float*)X  , C  , 2*INC);
	sIUpdate1(K, fabs(tmp[1]), (float*)X+1, C+1, 2*INC);
}

