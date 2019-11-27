#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>
#include <fstream>

using namespace std;

#define c 0.15

/* CUDA stuff */
__global__ void i_computation(const int i_max, double *deviceOld, double *deviceCurrent, double *deviceNext) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id > 0 && thread_id < i_max - 1) {
        int i = thread_id;
        deviceNext[i] = 2 * deviceCurrent[i] - deviceOld[i] + c * 
                        (deviceCurrent[i - 1] - (2 * deviceCurrent[i] - 
                        deviceCurrent[i + 1]));
    }
}

/* Write array to file */
void file_write_double_array(const char *filename, double *array, int n) {
    ofstream myfile(filename);
    if (myfile.is_open()) {
        for(int count = 0; count < n; count++){
            myfile << array[count] << "\n";
        }
        myfile.close();
    }
    else {
        cout << "Unable to open file";
    }
}

/* Fill array using sin */
void fill(double *array, int offset, int range, double sample_start,
          double sample_end) {
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = sin(sample_start + i * dx);
    }
}

double *simulate(const int i_max, const int t_max, const int block_size,
    double *old_array, double *current_array, double *next_array){
        int t;
        double *temp;

        /* Initialize memory on GPU */
        double *deviceOld = NULL;
        double *deviceCurrent = NULL;
        double *deviceNext = NULL;

        cudaMalloc((void **) &deviceOld, i_max * sizeof(double));
        if (deviceOld == NULL) {
            cerr << "Could not allocate space for old_array on GPU" << endl;
            exit(1);
        }
        cudaMalloc((void **) &deviceCurrent, i_max * sizeof(double));
        if (deviceCurrent == NULL) {
            cudaFree(deviceOld);
            cerr << "Could not allocate space for current_array on GPU" << endl;
            exit(1);
        }
        cudaMalloc((void **) &deviceNext, i_max * sizeof(double));
        if (deviceNext == NULL) {
            cudaFree(deviceOld);
            cudaFree(deviceCurrent);
            cerr << "Could not allocate space for next_array on GPU" << endl;
            exit(1);
        }

        /* Copy data to the GPU */
        cudaMemcpy(deviceOld, old_array, i_max * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceCurrent, current_array, i_max * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceNext, next_array, i_max * sizeof(double), cudaMemcpyHostToDevice);

        /* Run the kernel */
        int n_blocks = i_max / block_size;
        if (i_max % block_size != 0) {
            n_blocks++;
        }
        for (t = 0; t < t_max; t++) {
            // Run one iteration
            i_computation<<<n_blocks, block_size>>>(i_max, deviceOld, deviceCurrent, deviceNext);
            cout << cudaGetErrorString(cudaGetLastError()) << endl;

            // Swap pointers
            temp = deviceOld;
            deviceOld = deviceCurrent;
            deviceCurrent = deviceNext;
            deviceNext = temp;
        }

        /* Copy data back to main memory */
        cudaMemcpy(old_array, deviceOld, i_max * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(current_array, deviceCurrent, i_max * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(next_array, deviceNext, i_max * sizeof(double), cudaMemcpyDeviceToHost);

        /* Dealloc the memory on the GPU */
        cudaFree(deviceOld);
        cudaFree(deviceCurrent);
        cudaFree(deviceNext);

        return current_array;
}

int main(int argc, char* argv[]) {
    const int i_max      = atoi(argv[1]);
    const int t_max      = atoi(argv[2]);
    const int block_size = atoi(argv[3]);
    
    timer waveTimer("wave timer");

    double *old_array = (double*) calloc(i_max, sizeof(double));
    double *current_array = (double*) calloc(i_max, sizeof(double));
    double *next_array = (double*) calloc(i_max, sizeof(double));
    double *result_array = (double*) calloc(i_max, sizeof(double));
    
    fill(old_array, 1, i_max/4, 0, 2*3.14);
    fill(current_array, 2, i_max/4, 0, 2*3.14);

    waveTimer.start();
    result_array = simulate(i_max, t_max, block_size, old_array, current_array, next_array);
    waveTimer.stop();

    cout << waveTimer;
    
    file_write_double_array("result.txt", result_array, i_max);

    return 0;
}
