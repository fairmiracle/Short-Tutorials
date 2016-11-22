/*
* A simple example to show BLAS is faster than you
*
* Complie command under Linux:
* gcc -c useBLAS.c -fPIC -std=c99
* gcc useBLAS.o -o useBLAS -lgsl -lgslcblas
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_cblas.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void naivemv(double *A, double *x, double *y, int m, int n);

int main(int argc, char const *argv[])
{
	int n = 10000;
	double *A = Malloc(double,n*n);
	double *x = Malloc(double,n);
	double *y1 = Malloc(double,n);
	double *y2 = Malloc(double,n);

	/* random generation */ 
	srand(time(NULL));
	for(int i = 0; i < n; ++i)
	{
		x[i] = (double)rand() / (double)RAND_MAX;
		for(int j = 0; j < n; ++j)
			A[i+j*n] = (double)rand() / (double)RAND_MAX;
	}

	/* call naive implementation */
	clock_t tic = clock();
	naivemv(A, x, y1, n, n);
	double cpu_time_used1 = (double)(clock() - tic) / CLOCKS_PER_SEC;
	printf("Naive cpu time: %f\n", cpu_time_used1);

	/* call CBLAS Level2 */
	tic = clock();
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1, A, n, x, 1, 0, y2, 1);
	double cpu_time_used2 = (double)(clock() - tic) / CLOCKS_PER_SEC;
	printf("CBLAS cpu time: %f\n", cpu_time_used2);

	double *diffy = Malloc(double,n);
	memcpy(diffy, y1, n*sizeof(double));
	cblas_daxpy(n, -1, y2, 1, diffy, 1);
	/* call CBLAS Level1 */
	printf("Gap ||y1-y2|| between two methods: %f\n", cblas_ddot(n, diffy, 1, diffy, 1));

	/* Clear memory */
	free(A);
	free(x);
	free(y1);
	free(y2);
	free(diffy);
	return EXIT_SUCCESS;
}

/* Naive implementation of matrix-vector production
*  y = A*x, as y_i = \sum_jA_{ij}*x_j
*/
void naivemv(double *A, double *x, double *y, int m, int n)
{
	for (int i = 0; i < m; ++i)
	{
		double rowSum = 0;
		for (int j = 0; j < n; ++j)
		{
			rowSum += A[i+j*n]*x[j];
		}
		y[i] = rowSum;
	}
}