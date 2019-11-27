#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/*
*   Implements a REAL modulo function
**/
char mod(char x, char N) {
    // Source: https://stackoverflow.com/questions/11720656/modulo-operation-with-negative-numbers
    return (x % N + N) %N;
}

/*
*   Implements a REAL modulo function but on the GPU
**/
__device__ char cudaMod(char x, char N) {
    // Source: https://stackoverflow.com/questions/11720656/modulo-operation-with-negative-numbers
    return (x % N + N) %N;
}

/**
 * Given a pointer to an integer array and the argv array, compute
 * the encryption key list into the int array and return how many
 * keys are given.
 */
int encryption_key_parse(char **argv, int **int_array) {
    int i = 0;

    /* Some loops to check if the brackets/comma's are present */
    while(1) {
        if(argv[3][i] == '\0')
            break;
        i++;
    }

    char list[i];
    for(int j = 0; j <= i; j++) {
        list[j] = argv[3][j];
    }

    if(list[0] != '[' || list[i - 1] != ']') {
        printf("Error: Forget the brackets ([ or ]) \n");
        return 0;
    }

    char list2[i - 2];
    char check;
    int num_keys = 1;
    for(int j = 0; j <= (i - 3); j++) {
        check = list[j+1];
        list2[j] = check;
        if(check == ',') {
            num_keys++;
        }
    }

    /* Malloc the keys for use in the main function */
    int *keys;
    keys = (int*) malloc(sizeof(int)*num_keys);
    char *pointer;
    int keycounter = 0;
    pointer = strtok(list2, ",");

    /* Go through the string */
    while(pointer != NULL) {
        int num = atoi(pointer);
        keys[keycounter] = num;
        keycounter++;

        /* Skip next comma */
        pointer = strtok(NULL, ",");
    }

    *int_array = keys;
    return num_keys;
}

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


__global__ void encryptKernel(char* deviceDataIn, char* deviceDataOut, int key_length, char *deviceKey) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    deviceDataOut[index] = cudaMod(deviceDataIn[index] - ' ' + deviceKey[0], 95) + ' ';
}

__global__ void decryptKernel(char* deviceDataIn, char* deviceDataOut, int key_length, char *deviceKey) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    deviceDataOut[index] = cudaMod(deviceDataIn[index] - ' ' - deviceKey[0], 95) + ' ';
}

int fileSize() {
  int size;

  ifstream file ("original.data", ios::in|ios::binary|ios::ate);
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

int EncryptSeq (int n, char* data_in, char* data_out, int key_length, int* key) 
{    
    int i;
    timer sequentialTime = timer("Sequential encryption");
    
    sequentialTime.start();
    for (i=0; i<n; i++) {
        // shift the letter according to caesars cypher
        data_out[i] = mod(data_in[i] - ' ' + *key,95) + ' ';
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;
    
    return 0; 
}

int DecryptSeq (int n, char* data_in, char* data_out, int key_length, int *key)
{
  int i;
  timer sequentialTime = timer("Sequential decryption");

  sequentialTime.start();
  for (i=0; i<n; i++) {
        // shift the letter according to caesars cypher
        data_out[i] = mod(data_in[i] - ' ' - *key,95) + ' ';
    }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

  return 0;
}


int EncryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char *deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        checkCudaCall(cudaFree(deviceDataOut));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length*sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    encryptKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceDataIn, deviceDataOut, key_length, deviceKey);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int DecryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char *deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        checkCudaCall(cudaFree(deviceDataOut));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length*sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    decryptKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceDataIn, deviceDataOut, key_length, deviceKey);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int main(int argc, char* argv[]) {
    int *enc_key;
    int key_length = 0;
    if (argc == 4){
        key_length = encryption_key_parse(argv, (int**)&enc_key);
    }

    int n;
    n = fileSize();
    if (n == -1) {
	      cout << "File not found! Exiting ... " << endl;
	      exit(0);
    }

    char* data_in = new char[n];
    char* data_out = new char[n];
    readData("original.data", data_in);

    cout << "Encrypting a file of " << n << " characters." << endl;

    EncryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential.data", data_out);
    EncryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "cuda.data", data_out);

    cout << "Decrypting a file of " << n << "characters" << endl;
    readData("sequential.data", data_in);
    DecryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential_decrypted.data", data_out);
    readData("cuda.data", data_in);
    DecryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "recovered.data", data_out);

    delete[] data_in;
    delete[] data_out;

    return 0;
}
