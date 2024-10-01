#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> //  to compare float values
#include <cassert> // For assertion

#include "gpuErrchk.h"

using namespace std;

constexpr size_t ELEMENTS = 2048;

__global__ void vecadd(const int *A, const int *B, int *C)
{
	// Get block index
	unsigned int block_idx = blockIdx.x;
	// Get thread index
	unsigned int thread_idx = threadIdx.x;
	// Get the number of threads per block
	unsigned int block_dim = blockDim.x;
	// Get the thread's unique ID - (block_idx * block_dim) + thread_idx;
	unsigned int idx = (block_idx * block_dim) + thread_idx;
	// Add corresponding locations of A and B and store in C
	C[idx] = A[idx] + B[idx];
}


__global__ void simple_multiply(float *output_C, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, const float *input_A, const float *input_B)
{
    // Get global position in Y direction
    unsigned int row = (blockIdx.y * 1024) + threadIdx.y;
    // Get global position in X direction
    unsigned int col = (blockIdx.x * 1024) + threadIdx.x;

 // check that the row and column are within matrix bounds
    if (row < height_A && col < width_B)
    {
        float sum = 0.0f;

        // Compute the dot product of row from A and column from B
        for (unsigned int i = 0; i < width_A; ++i)
        {
            sum += input_A[row * width_A + i] * input_B[i * width_B + col];
        }

        // Store the result in matrix C
        output_C[row * width_B + col] = sum;
    }
}

// Function to initialize matrix A and matrix B with test data
void initialize_matrices(float* A, float* B, unsigned int width_A, unsigned int height_A, unsigned int width_B)
{
    // Initialize matrix A with sequential values
    for (unsigned int i = 0; i < height_A; ++i)
    {
        for (unsigned int j = 0; j < width_A; ++j)
        {
            A[i * width_A + j] = static_cast<float>(i * width_A + j);
        }
    }

    // Initialize matrix B as identity matrix for simplicity
    for (unsigned int i = 0; i < width_B; ++i)
    {
        for (unsigned int j = 0; j < height_A; ++j)
        {
            B[i * height_A + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Helper function to perform matrix multiplication on CPU for verification
void cpu_matrix_multiply(const float* A, const float* B, float* C, unsigned int width_A, unsigned int height_A, unsigned int width_B)
{
    for (unsigned int row = 0; row < height_A; ++row)
    {
        for (unsigned int col = 0; col < width_B; ++col)
        {
            float sum = 0.0f;
            for (unsigned int k = 0; k < width_A; ++k)
            {
                sum += A[row * width_A + k] * B[k * width_B + col];
            }
            C[row * width_B + col] = sum;
        }
    }
}

// Function to compare GPU and CPU results
void compare_results(const float* gpu_result, const float* cpu_result, unsigned int size)
{
    bool correct = true;
    for (unsigned int i = 0; i < size; ++i)
    {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5) // Compare with a small tolerance
        {
            std::cerr << "Mismatch at index " << i << ": GPU result " << gpu_result[i] << ", CPU result " << cpu_result[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct)
        std::cout << "Matrix multiplication test PASSED!" << std::endl;
    else
        std::cout << "Matrix multiplication test FAILED!" << std::endl;
}

// Function to print a matrix for debugging
void print_matrix(const float* matrix, unsigned int rows, unsigned int cols)
{
    for (unsigned int i = 0; i < rows; ++i)
    {
        for (unsigned int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char **argv)
{
	// Create host memory
	auto data_size = sizeof(int) * ELEMENTS;
	vector<int> A(ELEMENTS);    // Input aray
	vector<int> B(ELEMENTS);    // Input array
	vector<int> C(ELEMENTS);    // Output array

	// Initialise input data
	for (unsigned int i = 0; i < ELEMENTS; ++i)
		A[i] = B[i] = i;

	// Declare buffers
	int *buffer_A, *buffer_B, *buffer_C;

	// Initialise buffers
	cudaMalloc((void**)&buffer_A, data_size);
	cudaMalloc((void**)&buffer_B, data_size);
	cudaMalloc((void**)&buffer_C, data_size);

	// Write host data to device
	cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);


	// Run kernel with one thread for each element
	// First value is number of blocks, second is threads per block.  Max 1024 threads per block
	vecadd<<<ELEMENTS / 1024, 1024>>>(buffer_A, buffer_B, buffer_C);

	// Wait for kernel to complete
	cudaDeviceSynchronize();

	// Read output buffer back to the host
	cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);

	// Clean up resources
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);

	// Test that the results are correct
	for (int i = 0; i < 2048; ++i)
		if (C[i] != i + i)
			cout << "Error: " << i << endl;

	cout << "Finished" << endl;

	// Define matrix dimensions for matrix multiplication
    const unsigned int width_A = 32;   // Number of columns in matrix A
    const unsigned int height_A = 32;  // Number of rows in matrix A
    const unsigned int width_B = 32;   // Number of columns in matrix B
    const unsigned int height_B = 32;  // Number of rows in matrix B (must match width_A)

    // Allocate host memory for matrices A, B, and C
    size_t size_A = width_A * height_A * sizeof(float);
    size_t size_B = width_B * height_B * sizeof(float);
    size_t size_C = width_A * width_B * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    // Initialize matrices A and B with test data
    initialize_matrices(h_A, h_B, width_A, height_A, width_B);

    // Print matrices A and B (optional)
    std::cout << "Matrix A:" << std::endl;
    print_matrix(h_A, height_A, width_A);

    std::cout << "Matrix B:" << std::endl;
    print_matrix(h_B, height_B, width_B);

    // Allocate device memory for matrices A, B, and C
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy host matrices A and B to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define block dimensions and grid dimensions
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width_B + dimBlock.x - 1) / dimBlock.x,
                 (height_A + dimBlock.y - 1) / dimBlock.y);

    // Launch the matrix multiplication kernel
    simple_multiply<<<dimGrid, dimBlock>>>(d_C, width_A, height_A, width_B, height_B, d_A, d_B);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Copy the result matrix C back to the host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

     // Print GPU result matrix C
    std::cout << "Matrix C (Result from GPU):" << std::endl;
    print_matrix(h_C, height_A, width_B);

    // Allocate memory for CPU result
    float* h_C_cpu = (float*)malloc(size_C);

    // Perform matrix multiplication on CPU for verification
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, width_A, height_A, width_B);

    // Print CPU result matrix C
    std::cout << "Matrix C (Result from CPU):" << std::endl;
    print_matrix(h_C_cpu, height_A, width_B);

    // Compare GPU and CPU results
    compare_results(h_C, h_C_cpu, height_A * width_B);

    // Free CPU result memory
    free(h_C_cpu);

    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cout << "Finished matrix multiplication" << endl;

	return 0;
}

