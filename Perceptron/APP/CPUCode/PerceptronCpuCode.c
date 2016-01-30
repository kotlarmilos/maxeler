#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

double getTime(void);
void generate(int points,int maxIteration,float* arr_x1, float* arr_x2, float* arr_cls);
int getRand(int min_n, int max_n);
void PerceptronCPU(int points,int maxIteration,float* arr_x1, float* arr_x2, float* arr_cls,float *arr_y,float *w0,float *w1,float *w2);

int main(void)
{
	srand(time(NULL));
	const int points = 16;
	const int maxIteration = 100;

	int size=points*maxIteration;

	double startTime, cpuDuration, dfeDuration;

	float *arr_x1 = calloc(size, sizeof(float));
	float *arr_x2 = calloc(size, sizeof(float));
	float *arr_cls = calloc(size, sizeof(float));

	float *arr_y = calloc(size, sizeof(float));

	float *w0 = calloc(1, sizeof(float));
	float *w1 = calloc(1, sizeof(float));
	float *w2 = calloc(1, sizeof(float));

	float *w0_dfe = calloc(size, sizeof(float));
	float *w1_dfe = calloc(size, sizeof(float));
	float *w2_dfe = calloc(size, sizeof(float));

	printf("Generating input data...\n");
	generate(points,maxIteration,arr_x1, arr_x2, arr_cls);

	*w0 = 0;
	*w1 = 0;
	*w2 = 0;

	printf("Running on CPU...\n");
	startTime = getTime();
	PerceptronCPU(points,maxIteration,arr_x1,arr_x2,arr_cls,arr_y,w0,w1,w2);
	cpuDuration = getTime() - startTime;

	printf("w0: %g\n",*w0);
	printf("w1: %g\n",*w1);
	printf("w2: %g\n",*w2);

	printf("\n\nRunning on DFE...\n");
	startTime = getTime();
	Perceptron(size,arr_x1,arr_x2,arr_cls,arr_y,w0_dfe,w1_dfe,w2_dfe);
	dfeDuration = getTime() - startTime;

	printf("w0: %g\n",w0_dfe[size-1]);
	printf("w1: %g\n",w1_dfe[size-1]);
	printf("w2: %g\n",w2_dfe[size-1]);

	printf("\n\nCPU compute time: %g s\n", cpuDuration);
	printf("DFE compute time: %g s\n", dfeDuration);

	if (*w0==w0_dfe[size-1] && *w1==w1_dfe[size-1] && *w2==w2_dfe[size-1]){
		printf("Test passed OK!");
	}else{
		printf("Test failed!");
	}

	free(arr_x1);
	free(arr_x2);
	free(arr_cls);
	free(arr_y);

	free(w0);
	free(w1);
	free(w2);

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

void generate(int points,int maxIteration,float* arr_x1, float* arr_x2, float* arr_cls){
	int Ax = getRand(-points, points);
	int Ay = getRand(-points, points);

	int Bx = getRand(-points, points);
	int By = getRand(-points, points);

	for (int i = 0; i < points; i++){
		arr_x1[i] = getRand(-points, points);
		arr_x2[i] = getRand(-points, points);
		int side = (Bx - Ax) * (arr_x2[i] - Ay) - (By - Ay) * (arr_x1[i] - Ax);
		if (side > 0)
			arr_cls[i] = 1;
		else
			arr_cls[i] = -1;

		for (int j=1;j<maxIteration;j++){
			arr_x1[j*points+i] = arr_x1[i];
			arr_x2[j*points+i] = arr_x2[i];

			arr_cls[j*points+i]=arr_cls[i];
		}

		if (points<=16)
			printf("(%g,%g,%g)\n",arr_x1[i],arr_x2[i],arr_cls[i]);
	}
}

void PerceptronCPU(int points,int maxIteration,float* arr_x1, float* arr_x2, float* arr_cls,float *arr_y,float *w0,float *w1,float *w2){

	for (int i = 0; i < points*maxIteration; i++)
	{
		float x1 = arr_x1[i];
		float x2 = arr_x2[i];
		float y;

		if (((*w1 * x1) + (*w2 * x2) - *w0) < 0)
		{
			y = -1;
		}
		else
		{
			y = 1;
		}

		arr_y[i]=y;

		*w0 = *w0 + 0.5 * (arr_cls[i] - y) * (-1) / 2;
		*w1 = *w1 + 0.5 * (arr_cls[i] - y) * x1 / 2;
		*w2 = *w2 + 0.5 * (arr_cls[i] - y) * x2 / 2;
	}
}
