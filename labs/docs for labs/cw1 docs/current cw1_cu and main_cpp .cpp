#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cstring>  // For strcmp (used on the host side)
#include <algorithm> // For std::transform (used on the host side)


using namespace std;

// A device function to compare two strings for equality
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
        // Use custom GPU function to compare token with the substring in data
        if (gpu_strncmp(&data[idx], token, tokenLen)) {
            // Check if the match is a standalone word
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
    //chosen file
    const char *filepath = "C:/set10108-cw/set10108/labs/cw1/dataset/crime_and_punishment.txt";
    ifstream file(filepath, ios::binary);

    if (!file) {
        cerr << "Error: Could not open the file " << filepath << endl;
        return -1;
    }

    // Read file into buffer
    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    char* fileData = new char[fileSize]; // Allocate a raw char array
    if (!file.read(fileData, fileSize)) {
        cerr << "Error: Could not read the file content." << endl;
        delete[] fileData; // Free allocated memory
        return -1;
    }

    // Convert to lowercase using a loop instead of std::transform
    for (int i = 0; i < fileSize; ++i) {
        fileData[i] = tolower(fileData[i]);
    }

    // Define an array of tokens to search for
    const char* tokens[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};
    int numTokens = sizeof(tokens) / sizeof(tokens[0]);

    // Add header for GPU results in the console
    cout << "GPU Results:" << endl;  // Console output to indicate GPU results

    // Allocate memory on GPU for the data buffer
    char *d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, fileSize);
    if (err != cudaSuccess) {
        cerr << "Error allocating device memory for data: " << cudaGetErrorString(err) << endl;
        delete[] fileData;
        return -1;
    }
    cudaMemcpy(d_data, fileData, fileSize, cudaMemcpyHostToDevice);

       // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Loop through each token
    for (int i = 0; i < numTokens; ++i) {
        const char *token = tokens[i];
        int tokenLen = strlen(token);

        // Allocate memory on GPU for the token
        char *d_token;
        err = cudaMalloc((void**)&d_token, tokenLen);
        if (err != cudaSuccess) {
            cerr << "Error allocating device memory for token '" << token << "': " << cudaGetErrorString(err) << endl;
            continue; // Skip this word if allocation fails
        }
        cudaMemcpy(d_token, token, tokenLen, cudaMemcpyHostToDevice);

        // Allocate memory for the occurrences count
        int numOccurrences = 0;
        int *d_numOccurrences;
        err = cudaMalloc((void**)&d_numOccurrences, sizeof(int));
        if (err != cudaSuccess) {
            cerr << "Error allocating device memory for numOccurrences for token '" << token << "': " << cudaGetErrorString(err) << endl;
            cudaFree(d_token);
            continue; // Skip this word if allocation fails
        }
        cudaMemcpy(d_numOccurrences, &numOccurrences, sizeof(int), cudaMemcpyHostToDevice);

          // Record start event
        cudaEventRecord(start);

        // Launch kernel
        int blockSize = 256;      //256
        int gridSize = (fileSize + blockSize - 1) / blockSize;
        calc_token_occurrences_kernel<<<gridSize, blockSize>>>(d_data, fileSize, d_token, tokenLen, d_numOccurrences);

        // Record stop event and calculate elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Synchronize and check for errors after kernel launch
        cudaError_t errSync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            cerr << "Error during kernel execution for token '" << token << "': " << cudaGetErrorString(errSync) << endl;
            cudaFree(d_token);
            cudaFree(d_numOccurrences);
            continue; // Skip this word if an error occurs
        }

        // Copy result back to host
        cudaMemcpy(&numOccurrences, d_numOccurrences, sizeof(int), cudaMemcpyDeviceToHost);

       // Print debug statement to ensure token has been processed
        cout << "Processed token '" << token << "' with occurrences: " << numOccurrences << " in " << milliseconds << " ms" << endl;

        // Free memory allocated for token and occurrences count
        cudaFree(d_token);
        cudaFree(d_numOccurrences);
    }

    // Free memory
    cudaFree(d_data);
    delete[] fileData; // Free allocated memory

    return 0;
}
---------------------------------------------------------------------------------------
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>

//to do list
//check the text file for improvements

using namespace std;
using namespace std::chrono;

std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i< int(data.size()); ++i)
    {
        // test 1: does this match the token?
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the prefix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

int main()
{
    // Example chosen file
    const char * filepath = "C:/set10108-cw/set10108/labs/cw1/dataset/crime_and_punishment.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Example word list
    const char * words[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};
    std::cout << "CPU Results:" << endl;
    for(const char * word : words)
    {
        auto start = high_resolution_clock::now(); // Start timing

        int occurrences = calc_token_occurrences(file_data, word);

        auto stop = high_resolution_clock::now(); // Stop timing
        auto duration = duration_cast<milliseconds>(stop - start);

        std::cout << "Found " << occurrences << " occurrences of word: " << word << " in " << duration.count() << " ms" << std::endl;
    }
    //----------------------------------------------------------------------------------------
    
    
 
    // Run the CUDA program to generate the results
    int retCode = system("C:/set10108-cw/set10108/labs/cw1/build/Debug/cw1-cuda.exe"); 
    if (retCode != 0) {
        cerr << "Error: Failed to run the CUDA program." << endl;
        return -1;
    }



    return 0;
}