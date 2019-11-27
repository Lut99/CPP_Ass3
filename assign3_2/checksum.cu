#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

int fileSize(char *fileName) {
  int size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.close();
  }
  else {
    cout << "Unable to open file";
    size = -1;
  }
  return size;
}

int readData(char *fileName, char *data) {

  streampos size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.seekg (0, ios::beg);
    file.read (data, size);
    file.close();

    cout << "The entire file content is in memory." << endl;
  }
  else cout << "Unable to open file" << endl;
  return 0;
}

int writeData(int size, char *fileName, char *data) {
  ofstream file (fileName, ios::out|ios::binary|ios::trunc);
  if (file.is_open())
  {
    file.write (data, size);
    file.close();

    cout << "The entire file content was written to file." << endl;
    return 0;
  }
  else cout << "Unable to open file";

  return -1;
}

 __global__ void checksumKernel(unsigned int *deviceDataIn){
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
}

unsigned int checksumSeq (int n, unsigned int* data_in) {
    int i;
    timer sequentialTime = timer("Sequential checksum");

    sequentialTime.start();
    for (i=0; i<n; i++) {}
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Checksum (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}

/**
 * The checksumCuda handler that initialises the arrays to be used and calls
 * the checksum kernel. It also computes the missing values not calculated
 * on the GPU. It then adds all values together and prints the checksum
 */
 unsigned int checksumCuda (int n, unsigned int* data_in) {
    int threadBlockSize = 512;

    /* allocate the vectors on the GPU */
    unsigned int* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(unsigned int)));
    if (deviceDataIn == NULL) {
            cout << "could not allocate memory!" << endl;
            exit(1);
    }

    timer kernelTime  = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    /* copy the original vectors to the GPU */
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    kernelTime.start();
    checksumKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceDataIn);
    cudaDeviceSynchronize();
    kernelTime.stop();

    // Copies back the correct data
    checkCudaCall(cudaMemcpy(data_in, deviceDataIn, n*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* check whether the kernel invocation was successful */
    checkCudaCall(cudaGetLastError());

    /* copy result back */
    checkCudaCall(cudaFree(deviceDataIn));

    /* The times are printed */
    cout << fixed << setprecision(6);
    cout << "Kernel: \t\t" << kernelTime.getElapsed() << " seconds." << endl;
    cout << "Memory: \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return 0;
}

int main(int argc, char* argv[]) {
    int n;
    int seq;
    char *fileName;

    // Arg parse
    if (argc == 3) {
        fileName = (char*)argv[2];
        seq = atoi(argv[1]);

        printf("Chosen for option: %d\n", seq);
        printf("opening file %s\n", fileName);
    } else {
        printf("non valid options\n");
        return EXIT_FAILURE;
    }
    n = fileSize(fileName);
    if (n == -1) {
        printf("file not found\n");
        exit(0);
    }

    char* data_in = new char[n];
    readData(fileName, data_in);
    unsigned int *data_in_raw = new unsigned int[n];
    for (int i = 0; i < n; i++){
        data_in_raw[i] = data_in[i];
    }

    /* Check the option to determine the functions to be called */
    if (seq == 1){
        // Only sequential checkusm is ran
        unsigned int checksum = checksumSeq(n, data_in_raw);
        printf("Sequental checksum %u\n", checksum);
    } else if (seq == 0) {
        // Only cuda checksum is ran
        unsigned int checksum = checksumCuda(n, data_in_raw);
        printf("Cuda checksum %u\n", checksum);
    } else if (seq == 2){
        // Both the sequential and the cuda checksum are run
        unsigned int checksum = checksumCuda(n, data_in_raw);
        printf("Cuda checksum %u\n", checksum);
        checksum = checksumSeq(n, data_in_raw);
        printf("Sequental checksum %u\n", checksum);
    }

    delete[] data_in;
    return 0;
}