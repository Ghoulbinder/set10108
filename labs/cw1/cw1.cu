#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cstring>  // For strcmp (used on the host side)
#include <algorithm> // For std::transform (used on the host side)


using namespace std;

__device__ bool gpu_strncmp(const char *str1, const char *str2, int n) {
    for (int i = 0; i < n; ++i) {
        if (str1[i] != str2[i]) {
            return false;
        }
    }
    return true;
}

__global__ void calc_token_occurrences_kernel(char *data, int dataSize, char *token, int tokenLen, int *numOccurrences)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize - tokenLen) {
        if (gpu_strncmp(&data[idx], token, tokenLen)) {
            bool validPrefix = (idx == 0) || (data[idx - 1] < 'a' || data[idx - 1] > 'z');
            bool validSuffix = (idx + tokenLen >= dataSize) || (data[idx + tokenLen] < 'a' || data[idx + tokenLen] > 'z');
            if (validPrefix && validSuffix) {
                atomicAdd(numOccurrences, 1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char* filepath = "C:/set10108-cw/set10108/labs/cw1/dataset/shakespeare.txt";
    ifstream file(filepath, ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open the file " << filepath << std::endl;
        return -1;
    }

    // Read file into buffer
    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    char* fileData = new char[fileSize]; // Allocate a raw char array
    if (!file.read(fileData, fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        delete[] fileData; // Free allocated memory
        return -1;
    }

    // Convert to lowercase using a loop
    for (int i = 0; i < fileSize; ++i) {
        fileData[i] = tolower(fileData[i]);
    }

    const char* tokens[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};
    int numTokens = sizeof(tokens) / sizeof(tokens[0]);

     // Add header for GPU results in the console
    cout << "GPU Results:" << endl;  // Console output to indicate GPU results

    // Allocate memory on GPU for the data buffer
    char *d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, fileSize);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for data: " << cudaGetErrorString(err) << std::endl;
        delete[] fileData;
        return -1;
    }
    cudaMemcpy(d_data, fileData, fileSize, cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA Occupancy API to calculate maximum number of active blocks per SM
    int blockSize = 256;  // Block size (threads per block)
    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, calc_token_occurrences_kernel, blockSize, 0);
    std::cout << "Max active blocks per SM: " << numBlocksPerSM << std::endl;

    // Calculate Grid size
    int gridSize = (fileSize + blockSize - 1) / blockSize;

    // Loop through each token
    for (int i = 0; i < numTokens; ++i) {
        const char *token = tokens[i];
        int tokenLen = strlen(token);

        char *d_token;
        err = cudaMalloc((void**)&d_token, tokenLen);
        if (err != cudaSuccess) {
            std::cerr << "Error allocating device memory for token '" << token << "': " << cudaGetErrorString(err) << std::endl;
            continue; // Skip this word if allocation fails
        }
        cudaMemcpy(d_token, token, tokenLen, cudaMemcpyHostToDevice);

        int numOccurrences = 0;
        int *d_numOccurrences;
        err = cudaMalloc((void**)&d_numOccurrences, sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating device memory for numOccurrences for token '" << token << "': " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_token);
            continue; // Skip this word if allocation fails
        }
        cudaMemcpy(d_numOccurrences, &numOccurrences, sizeof(int), cudaMemcpyHostToDevice);

        // Record start event
        cudaEventRecord(start);

        // Launch kernel
        calc_token_occurrences_kernel<<<gridSize, blockSize>>>(d_data, fileSize, d_token, tokenLen, d_numOccurrences);

        // Record stop event and calculate elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaError_t errSync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "Error during kernel execution for token '" << token << "': " << cudaGetErrorString(errSync) << std::endl;
            cudaFree(d_token);
            cudaFree(d_numOccurrences);
            continue; // Skip this word if an error occurs
        }

        // Copy result back to host
        cudaMemcpy(&numOccurrences, d_numOccurrences, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Processed token '" << token << "' with occurrences: " << numOccurrences << " in " << milliseconds << " ms" << std::endl;

        cudaFree(d_token);
        cudaFree(d_numOccurrences);
    }

    cudaFree(d_data);
    delete[] fileData;

    return 0;
}
