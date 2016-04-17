#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

double getTime(void);
void generate(int size,double* mappedRom_x1, double* mappedRom_x2, double* mappedRom_class);
int getRand(int min_n, int max_n);
void PerceptronCPU(int size,int iteration,double* mappedRom_x1, double* mappedRom_x2, double* mappedRom_class,float alpha, float bias,float *w0_cpu,float *w1_cpu,float *w2_cpu);

int main(void)
{
	srand(time(NULL));

	const int size = 16;
	const int iteration=100;

	double startTime, cpuDuration, dfeDuration;

	int sizeBytesFloat = iteration * sizeof(float);
	int sizeBytesDouble = size * sizeof(double);

	double *mappedRom_x1=malloc(sizeBytesDouble);
	double *mappedRom_x2=malloc(sizeBytesDouble);
	double *mappedRom_class=malloc(sizeBytesDouble);

	float *w0_cpu = malloc(sizeof(float));
	float *w1_cpu = malloc(sizeof(float));
	float *w2_cpu = malloc(sizeof(float));

	float *w0_dfe = malloc(sizeBytesFloat);
	float *w1_dfe = malloc(sizeBytesFloat);
	float *w2_dfe = malloc(sizeBytesFloat);

	float alpha=0.5;
	float bias=-1;

	printf("Generating input data...\n");
	generate(size,mappedRom_x1,mappedRom_x2,mappedRom_class);
	printf("Done.\n");

	printf("Running on CPU...\n");
	startTime = getTime();
	PerceptronCPU(size,iteration,mappedRom_x1,mappedRom_x2,mappedRom_class,alpha,bias,w0_cpu,w1_cpu,w2_cpu);
	cpuDuration = getTime() - startTime;
	printf("Done.\n");

	printf("Running on DFE...\n");
	startTime = getTime();
	PerceptronDFE(iteration, alpha,bias,w0_dfe,w1_dfe,w2_dfe,mappedRom_x1,mappedRom_x2,mappedRom_class);
	dfeDuration = getTime() - startTime;
	printf("Done.\n");

	printf("CPU: (%g, %g, %g)\n",*w0_cpu,*w1_cpu,*w2_cpu);
	printf("DFE: (%g, %g, %g)\n",w0_dfe[iteration-1],w1_dfe[iteration-1],w2_dfe[iteration-1]);

	printf("--\n");
	printf("Size: %d (%d B)\n",size, size*8*3);
	printf("Iterations: %d\n",iteration);
	printf("CPU compute time: %g s\n", cpuDuration);
	printf("DFE compute time: %g s\n", dfeDuration);

	printf("Result: ");
	if (*w0_cpu==w0_dfe[iteration-1] &&
		*w1_cpu==w1_dfe[iteration-1] &&
		*w2_cpu==w2_dfe[iteration-1]){
		printf("Test passed!\n");
	}else{
		printf("Test failed!***\n");
	}
	printf("--\n");

	free(mappedRom_x1);
	free(mappedRom_x2);
	free(mappedRom_class);

	free(w0_cpu);
	free(w1_cpu);
	free(w2_cpu);

	free(w0_dfe);
	free(w1_dfe);
	free(w2_dfe);
	return 0;
}


double getTime(void) {
	struct timeval time;
	gettimeofday(&time, NULL);
	return time.tv_sec + 1e-6 * time.tv_usec;
}

int getRand(int min_n, int max_n){
	return rand() % (max_n - min_n + 1) + min_n;
}

void generate(int size,double* mappedRom_x1, double* mappedRom_x2, double* mappedRom_class){
	int Ax = getRand(-size, size);
	int Ay = getRand(-size, size);

	int Bx = getRand(-size, size);
	int By = getRand(-size, size);

	for (int i = 0; i < size; i++){
		mappedRom_x1[i] = getRand(-size, size);
		mappedRom_x2[i] = getRand(-size, size);
		int side = (Bx - Ax) * (mappedRom_x2[i] - Ay) - (By - Ay) * (mappedRom_x1[i] - Ax);
		if (side > 0)
			mappedRom_class[i] = 1;
		else
			mappedRom_class[i] = -1;

		if (size<=16)
			printf("(%g,%g,%g)\n",mappedRom_x1[i],mappedRom_x2[i],mappedRom_class[i]);
	}
}

void PerceptronCPU(int size,int iteration,double* mappedRom_x1, double* mappedRom_x2, double* mappedRom_class,float alpha, float bias,float *w0_cpu,float *w1_cpu,float *w2_cpu){
	int iterations=0;

	float *w0_arr=malloc(size*sizeof(float));
	float *w1_arr=malloc(size*sizeof(float));
	float *w2_arr=malloc(size*sizeof(float));

	while (iterations<iteration){
		for (int i = 0; i < size; i++)
		{
			float x1 = mappedRom_x1[i];
			float x2 = mappedRom_x2[i];
			float y;

			if (((*w1_cpu * x1) + (*w2_cpu * x2) - *w0_cpu) < 0)
			{
				y = -1;
			}
			else
			{
				y = 1;
			}

			float w0_temp = alpha * (mappedRom_class[i] - y) * bias / 2;
			float w1_temp = alpha * (mappedRom_class[i] - y) * x1 / 2;
			float w2_temp = alpha * (mappedRom_class[i] - y) * x2 / 2;

			w0_arr[i] = w0_temp;
			w1_arr[i] = w1_temp;
			w2_arr[i] = w2_temp;
		}

		for (int i=0;i<size;i++){
			*w0_cpu += w0_arr[i];
			*w1_cpu += w1_arr[i];
			*w2_cpu += w2_arr[i];
		}

		iterations++;
	}
}

