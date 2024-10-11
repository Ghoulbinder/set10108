Using a `16x16` block size (which equals 256 threads) is a common practice in CUDA programming because it balances thread usage efficiency, memory utilization, and computational performance on most modern GPUs. The choice of block size directly impacts the performance and resource utilization of the GPU. Let's dive into more detail on why `16x16` is often chosen:

### 1. **Understanding CUDA Block and Grid Size**
CUDA defines its execution configuration in terms of blocks and grids:
- **Threads**: The smallest unit of execution in CUDA.
- **Blocks**: A group of threads that execute the same code. Threads within a block can communicate via shared memory and can synchronize.
- **Grids**: A group of blocks that execute the same kernel. Blocks within a grid operate independently of each other.

CUDA blocks can be defined in 1D, 2D, or 3D. The number of threads in a block (`blockDim`) and the number of blocks in a grid (`gridDim`) are critical for performance optimization.

### 2. **Why Choose 16x16 Block Size?**

Choosing a `16x16` block size means you have:
- **16 threads in the X dimension**.
- **16 threads in the Y dimension**.

This gives a total of `16 * 16 = 256 threads` in each block. Here’s why this configuration is often optimal:

#### a. **Warps and Efficient Thread Management**
- **Warp Size**: The fundamental execution unit of a CUDA-capable GPU is a *warp*. A warp is a group of 32 threads that execute instructions in lockstep.
- Using 256 threads means you have exactly **8 warps** per block (`256 / 32 = 8 warps`).
- This fits well with the warp scheduling mechanism of the GPU, ensuring that all threads in a block can be scheduled and managed without wasting resources.
  
#### b. **Occupancy and Resource Utilization**
- GPU performance is often limited by the number of active threads and blocks that can be maintained concurrently. This is known as *occupancy*.
- Occupancy depends on the number of registers and shared memory per block, and block size.
- A block with 256 threads balances the use of registers and shared memory, allowing the GPU to keep many blocks active at the same time. Higher occupancy usually translates to better performance, as the GPU can hide latencies better.

#### c. **Shared Memory and Memory Coalescing**
- A `16x16` block size results in good memory access patterns and coalescing for many algorithms, including matrix multiplications, stencil operations, and image processing.
- Each block of threads can operate on a `16x16` tile of data, which fits nicely in the GPU’s shared memory, reducing global memory access overhead and increasing bandwidth utilization.

#### d. **Spatial Data Locality**
- In 2D problems (like matrix operations or image processing), a `16x16` block maps naturally onto a 2D space, providing good spatial locality.
- This allows threads in a block to operate on data that is spatially close, minimizing global memory access and making it easier to implement efficient shared memory use.

### 3. **Comparing to Other Block Sizes**

The block size can vary based on the specific algorithm and hardware characteristics. Here's a quick comparison of different block sizes and when they might be appropriate:

- **`32x32` (1024 threads)**:
  - This is the maximum number of threads per block for most modern GPUs.
  - Best for compute-heavy kernels with minimal shared memory requirements.
  - Can reduce block count, which may reduce flexibility in scheduling.

- **`8x8` (64 threads)**:
  - Good for GPUs with fewer registers or shared memory.
  - May lead to underutilization of GPU resources as fewer threads are launched.

- **`16x16` (256 threads)**:
  - Balances between good occupancy and resource utilization.
  - Works well for a variety of problems, from general-purpose computation to image processing.

### 4. **When Not to Use `16x16`**
- If your problem is purely 1D, like a linear vector operation, a 1D block (e.g., 256 threads) might be more appropriate.
- If you are working with a very large dataset and have strict memory limitations, reducing the block size (e.g., `8x8`) could help to minimize shared memory usage and fit more blocks concurrently.
- For compute-intensive tasks that require many registers, reducing the block size can also be beneficial, as it allows more blocks to run concurrently.

### 5. **Performance Testing and Tuning**
While `16x16` is a good starting point, the best block size ultimately depends on your specific hardware and problem characteristics. It is recommended to:
- **Experiment** with different block sizes (e.g., `8x8`, `16x16`, `32x32`).
- **Profile** the performance using tools like *NVIDIA Nsight* to analyze resource utilization and kernel execution times.
- **Optimize** the block size based on the number of threads per multiprocessor, shared memory usage, and register count.

### Summary

Using a `16x16` block size provides a balanced configuration that:
- Maps well to the GPU’s warp size.
- Ensures good utilization of resources.
- Provides good spatial locality for 2D data operations.
- Helps maintain high occupancy and minimize memory access overheads.

This block size is suitable for most general-purpose and 2D problems, making it a popular choice for CUDA development. However, it’s always a good idea to experiment and profile with your specific application to find the optimal configuration.

No, in the previous setup, we haven't fully utilized the `grid` structure. We only defined `blocks` and `threads` within the kernel launch, but the `grid` layout has not been explicitly structured beyond a single dimension. CUDA uses a hierarchy of grids, blocks, and threads to organize parallel execution.

In CUDA, a grid is composed of blocks, and each block is composed of threads. While the code uses a one-dimensional grid (`blocksPerGrid`) and one-dimensional block (`threadsPerBlock`), we haven't yet demonstrated how to configure a 2D or 3D grid.

### Explanation of Grids and Blocks

1. **Threads**: The smallest unit of execution. A block is composed of many threads.
2. **Blocks**: A group of threads that execute the same code. They can communicate through shared memory.
3. **Grid**: A group of blocks that execute the same kernel. Each block can be indexed using its position in the grid.

### Modifying the `kernel.cu` to Use a Grid
Let's modify the `kernel.cu` file to explicitly define and use a 2D grid for executing the `countTokenOccurrencesKernel`. This will help us better organize and scale the search for larger datasets.

### Updated `kernel.cu` with Grid Usage

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>  // For strcmp and strlen

#include "gpuErrchk.h"  // Custom header for CUDA error checking

// CUDA kernel for counting token occurrences in the text using a 2D grid
__global__ void countTokenOccurrencesKernel(const char* data, int dataSize, const char* token, int tokenLen, int* result) {
    // Calculate the unique thread index within the grid using a 2D layout
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * blockDim.x * gridDim.x + col;

    if (idx + tokenLen <= dataSize) {
        // Compare substring starting at idx with token
        bool match = true;
        for (int i = 0; i < tokenLen; ++i) {
            if (data[idx + i] != token[i]) {
                match = false;
                break;
            }
        }

        // Use atomic addition to update the global result count if match is found
        if (match) {
            if ((idx == 0 || !isalpha(data[idx - 1])) && (idx + tokenLen == dataSize || !isalpha(data[idx + tokenLen]))) {
                atomicAdd(result, 1);
            }
        }
    }
}

// GPU Utility Class for managing GPU operations
class GpuTextSearcher {
public:
    GpuTextSearcher() : d_data(nullptr), d_result(nullptr), dataSize(0) {}

    // Allocate memory on the GPU and copy data to the GPU
    void allocateAndTransferData(const std::vector<char>& fileData) {
        dataSize = fileData.size();
        gpuErrchk(cudaMalloc((void**)&d_data, dataSize * sizeof(char)));
        gpuErrchk(cudaMemcpy(d_data, fileData.data(), dataSize * sizeof(char), cudaMemcpyHostToDevice));
    }

    // Allocate memory for the result on the GPU
    void allocateResultMemory() {
        gpuErrchk(cudaMalloc((void**)&d_result, sizeof(int)));
    }

    // Launch the CUDA kernel for text search
    void launchKernel(const char* token) {
        int tokenLen = strlen(token);
        int h_result = 0;

        // Initialize the result memory on the GPU
        gpuErrchk(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));

        // Configure a 2D grid and 2D block size
        dim3 threadsPerBlock(16, 16);  // 16 x 16 threads per block = 256 threads
        dim3 blocksPerGrid((dataSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (dataSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the 2D grid and block kernel
        countTokenOccurrencesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, dataSize, token, tokenLen, d_result);

        // Synchronize to ensure kernel execution is complete
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Retrieve the result from the GPU to the host
    int getResult() {
        int h_result;
        gpuErrchk(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        return h_result;
    }

    // Free allocated GPU memory
    void freeMemory() {
        if (d_data) cudaFree(d_data);
        if (d_result) cudaFree(d_result);
    }

    ~GpuTextSearcher() {
        freeMemory();
    }

private:
    char* d_data;
    int* d_result;
    int dataSize;
};

// Wrapper function for main.cpp to use the GPU functionality
extern "C" int gpu_count_token_occurrences(const std::vector<char>& fileData, const char* token) {
    GpuTextSearcher gpuSearcher;

    // Allocate memory and transfer data to the GPU
    gpuSearcher.allocateAndTransferData(fileData);

    // Allocate memory for the result on the GPU
    gpuSearcher.allocateResultMemory();

    // Launch the kernel
    gpuSearcher.launchKernel(token);

    // Retrieve the result and free GPU memory
    int result = gpuSearcher.getResult();
    gpuSearcher.freeMemory();

    return result;
}
```

### Explanation of the Grid and Block Configuration
1. **Threads Per Block:**
   - We define a `dim3` variable called `threadsPerBlock` with `16` threads in both `x` and `y` dimensions, making a 16x16 block (256 threads).

2. **Blocks Per Grid:**
   - We define a `dim3` variable called `blocksPerGrid`, which divides the total number of data elements by the number of threads in each dimension. This configuration ensures that the entire dataset is covered by the grid of blocks.

3. **Kernel Index Calculation:**
   - In the kernel, we use both `blockIdx` (block index in the grid) and `threadIdx` (thread index in the block) to calculate the unique thread index in the 2D grid:
     ```cpp
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int idx = row * blockDim.x * gridDim.x + col;
     ```
   - This formula converts the 2D grid and block layout into a unique 1D index, which is then used to access the character data.

### Summary
- The updated `kernel.cu` file now utilizes a **2D grid** structure to perform the text search.
- This structure can help parallelize larger datasets more effectively and distribute the workload evenly across multiple threads and blocks.
- The `main.cpp` file remains unchanged as it calls the wrapper function `gpu_count_token_occurrences()`.

If you want to further customize the grid configuration or explore 3D grids for more complex scenarios, you can modify the `dim3` variables and index calculations in the kernel accordingly. Let me know if you need further assistance!


Got it! We'll keep as much of the GPU-related code (such as memory allocations, data transfers, and kernel launches) in the `.cu` file, leaving `main.cpp` to handle only high-level logic and CPU operations. This will help modularize the code and keep the GPU-specific operations separate from the CPU code.

Here's how we can reorganize the project:

1. **`main.cpp`**: This file will handle file reading, CPU-based operations, and calling the GPU functions from the `.cu` file.
2. **`kernel.cu`**: This file will handle all the CUDA-related operations, including:
   - Memory allocations for the GPU.
   - Copying data to/from the GPU.
   - Launching the kernel for token search.
   - Cleaning up GPU memory after processing.

We'll define a GPU class or a set of GPU utility functions in the `.cu` file to handle all the GPU operations.

### Step 1: `kernel.cu` - Implement the GPU Functions
We'll create a `GpuTextSearcher` class in `kernel.cu` to encapsulate all the CUDA-related logic.

**`kernel.cu`:**
```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>  // For strcmp and strlen

#include "gpuErrchk.h"  // Custom header for CUDA error checking

// CUDA kernel for counting token occurrences in the text
__global__ void countTokenOccurrencesKernel(const char* data, int dataSize, const char* token, int tokenLen, int* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx + tokenLen <= dataSize) {
        // Compare substring starting at idx with token
        bool match = true;
        for (int i = 0; i < tokenLen; ++i) {
            if (data[idx + i] != token[i]) {
                match = false;
                break;
            }
        }

        // Use atomic addition to update the global result count if match is found
        if (match) {
            if ((idx == 0 || !isalpha(data[idx - 1])) && (idx + tokenLen == dataSize || !isalpha(data[idx + tokenLen]))) {
                atomicAdd(result, 1);
            }
        }
    }
}

// GPU Utility Class for managing GPU operations
class GpuTextSearcher {
public:
    GpuTextSearcher() : d_data(nullptr), d_result(nullptr), dataSize(0) {}

    // Allocate memory on the GPU and copy data to the GPU
    void allocateAndTransferData(const std::vector<char>& fileData) {
        dataSize = fileData.size();
        gpuErrchk(cudaMalloc((void**)&d_data, dataSize * sizeof(char)));
        gpuErrchk(cudaMemcpy(d_data, fileData.data(), dataSize * sizeof(char), cudaMemcpyHostToDevice));
    }

    // Allocate memory for the result on the GPU
    void allocateResultMemory() {
        gpuErrchk(cudaMalloc((void**)&d_result, sizeof(int)));
    }

    // Launch the CUDA kernel for text search
    void launchKernel(const char* token) {
        int tokenLen = strlen(token);
        int h_result = 0;

        // Initialize the result memory on the GPU
        gpuErrchk(cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice));

        // Configure kernel launch parameters
        int threadsPerBlock = 256;
        int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel
        countTokenOccurrencesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, dataSize, token, tokenLen, d_result);

        // Synchronize to ensure kernel execution is complete
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Retrieve the result from the GPU to the host
    int getResult() {
        int h_result;
        gpuErrchk(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        return h_result;
    }

    // Free allocated GPU memory
    void freeMemory() {
        if (d_data) cudaFree(d_data);
        if (d_result) cudaFree(d_result);
    }

    ~GpuTextSearcher() {
        freeMemory();
    }

private:
    char* d_data;
    int* d_result;
    int dataSize;
};

// Wrapper function for main.cpp to use the GPU functionality
extern "C" int gpu_count_token_occurrences(const std::vector<char>& fileData, const char* token) {
    GpuTextSearcher gpuSearcher;

    // Allocate memory and transfer data to the GPU
    gpuSearcher.allocateAndTransferData(fileData);

    // Allocate memory for the result on the GPU
    gpuSearcher.allocateResultMemory();

    // Launch the kernel
    gpuSearcher.launchKernel(token);

    // Retrieve the result and free GPU memory
    int result = gpuSearcher.getResult();
    gpuSearcher.freeMemory();

    return result;
}
```

### Step 2: `main.cpp` - High-Level Logic and Function Calls
Modify `main.cpp` to interact with the `GpuTextSearcher` class in `kernel.cu`. `main.cpp` will only handle file reading, CPU-based token counting, and calling the GPU-based functions.

**Updated `main.cpp`:**
```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// Function prototype for the GPU text search (defined in kernel.cu)
extern "C" int gpu_count_token_occurrences(const std::vector<char>& fileData, const char* token);

// Function to read the file and return the contents as a vector of characters
std::vector<char> read_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }
    file.close();

    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

// Function to count word occurrences using CPU (existing implementation)
int calc_token_occurrences(const std::vector<char>& data, const char* token) {
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i < int(data.size()); ++i) {
        if (strncmp(&data[i], token, tokenLen) != 0)
            continue;

        if (i - 1 >= 0 && std::isalpha(data[i - 1]))  // Prefix check
            continue;

        if (i + tokenLen < int(data.size()) && std::isalpha(data[i + tokenLen]))  // Suffix check
            continue;

        ++numOccurrences;
    }
    return numOccurrences;
}

int main() {
    const char* filepath = "C:/set10108-cw/set10108/labs/cw1/dataset/shakespeare.txt";
    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty()) return -1;

    // Word list to search for
    const char* words[] = {"sword", "fire", "death", "love", "hate", "the", "man", "woman"};
    for (const char* word : words) {
        // Perform GPU-based search and get the result
        int gpuOccurrences = gpu_count_token_occurrences(file_data, word);
        std::cout << "GPU Search: Found " << gpuOccurrences << " occurrences of word: " << word << std::endl;

        // Perform CPU-based search for comparison
        int cpuOccurrences = calc_token_occurrences(file_data, word);
        std::cout << "CPU Search: Found " << cpuOccurrences << " occurrences of word: " << word << std::endl;

        if (gpuOccurrences != cpuOccurrences) {
            std::cerr << "Mismatch in counts for word: " << word << " (CPU: " << cpuOccurrences << ", GPU: " << gpuOccurrences << ")" << std::endl;
        }
    }

    return 0;
}
```

### Step 3: Compile and Link the Project
To compile and link your project correctly, use the following commands or set up your IDE to include both `main.cpp` and `kernel.cu` files:

**Command-Line Compilation Example:**
```bash
nvcc -o text_search main.cpp kernel.cu -lcudart
```

### Summary
- **`kernel.cu`:** Manages all GPU-related operations using the `GpuTextSearcher` class.
- **`main.cpp`:** High-level logic to read files, handle CPU operations, and call GPU functions.

This approach keeps the GPU-related code separate, making `main.cpp` simpler and more focused on high-level program control. Let me know if you need further adjustments or have any questions!

