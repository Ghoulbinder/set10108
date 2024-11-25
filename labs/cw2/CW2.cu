#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cstring>  // For strcmp (used on the host side)
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;
using namespace std;

// Structure for RGBA pixel
struct rgba_t {
    uint8_t r, g, b, a;
};

// CUDA Kernel to calculate color temperature
__global__ void calculateTemperatures(const rgba_t* pixels, double* temperatures, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;

    rgba_t pixel = pixels[idx];

    // Normalize RGB values
    double red = pixel.r / 255.0;
    double green = pixel.g / 255.0;
    double blue = pixel.b / 255.0;

    // Gamma correction
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Chromaticity coordinates
    double sum = X + Y + Z;
    double x = (sum == 0) ? 0 : (X / sum);
    double y = (sum == 0) ? 0 : (Y / sum);

    // McCamy's formula for color temperature
    double n = (x - 0.332) / (y - 0.1858);
    temperatures[idx] = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;
}

// Helper function to load RGB data from an image file
std::vector<rgba_t> load_rgb(const char* filename, int& width, int& height) {
    int n;
    unsigned char* data = stbi_load(filename, &width, &height, &n, 4);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = reinterpret_cast<rgba_t*>(data);
    std::vector<rgba_t> vec(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// GPU-based function to calculate median temperature for an image
double calculateMedianTemperatureGPU(const std::vector<rgba_t>& image) {
    int totalPixels = image.size();

    // Allocate memory on the device
    rgba_t* d_pixels;
    double* d_temperatures;
    cudaMalloc(&d_pixels, totalPixels * sizeof(rgba_t));
    cudaMalloc(&d_temperatures, totalPixels * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_pixels, image.data(), totalPixels * sizeof(rgba_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (totalPixels + blockSize - 1) / blockSize;
    calculateTemperatures<<<gridSize, blockSize>>>(d_pixels, d_temperatures, totalPixels);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_pixels);
        cudaFree(d_temperatures);
        return 0;
    }

    cudaDeviceSynchronize();

    // Copy results back to the host
    std::vector<double> temperatures(totalPixels);
    cudaMemcpy(temperatures.data(), d_temperatures, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);

    // Sort and calculate median
    std::sort(temperatures.begin(), temperatures.end());
    double median = (totalPixels % 2 == 0)
        ? (temperatures[totalPixels / 2 - 1] + temperatures[totalPixels / 2]) / 2.0
        : temperatures[totalPixels / 2];

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_temperatures);

    return median;
}

// Exported function for GPU-based sorting
extern "C" void gpuSortImagesByTemperature(std::vector<std::string>& filenames) {
    std::vector<std::pair<std::string, double>> tempPairs;

    // Calculate median temperature for each image
    for (const auto& filename : filenames) {
        int width, height;
        auto image = load_rgb(filename.c_str(), width, height);
        if (!image.empty()) {
            double medianTemp = calculateMedianTemperatureGPU(image);
            tempPairs.emplace_back(filename, medianTemp);
        }
    }

    // Sort by temperature
    std::sort(tempPairs.begin(), tempPairs.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // Update filenames with sorted order
    for (size_t i = 0; i < tempPairs.size(); ++i) {
        filenames[i] = tempPairs[i].first;
    }

    // Print results
    for (const auto& pair : tempPairs) {
        std::cout << "Image: " << pair.first << " - Assigned Temperature: " << pair.second << "K\n";
    }
}
int main() {
    const char* image_folder = "C:/set10108-cw/set10108/labs/cw2/images/unsorted";
    if (!fs::is_directory(image_folder)) {
        std::cerr << "Directory not found: " << image_folder << std::endl;
        return -1;
    }

    // Collect image filenames
    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder)) {
        imageFilenames.push_back(p.path().u8string());
    }

    // Sort images using GPU
    auto startTime = std::chrono::high_resolution_clock::now();
    gpuSortImagesByTemperature(imageFilenames);
    auto endTime = std::chrono::high_resolution_clock::now();

    std::cout << "Sorting completed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
              << " ms.\n";

    return 0;
}

