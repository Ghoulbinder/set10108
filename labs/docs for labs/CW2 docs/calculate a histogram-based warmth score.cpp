// Calculate dominant color temperature for an image
double filename_to_dominant_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);

    int warmPixels = 0;
    int coolPixels = 0;

    for (const auto& pixel : rgbadata) {
        double temperature = daylightApproximationColorTemperature(pixel);
        if (temperature >= 5000.0) {  // Arbitrary warm threshold (5000K)
            warmPixels++;
        }
        else {
            coolPixels++;
        }
    }

    double warmRatio = static_cast<double>(warmPixels) / rgbadata.size();
    return warmRatio;
}
// Function to calculate a histogram-based warmth score
double calculateHistogramWarmthScore(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);

    // Initialize bins (adjust these thresholds as needed)
    const double warmThreshold1 = 3000;
    const double warmThreshold2 = 5000;
    const double coolThreshold1 = 7000;

    int warmCount = 0;
    int neutralCount = 0;
    int coolCount = 0;

    // Bin pixels into warm, neutral, and cool based on temperature
    for (const auto& pixel : rgbadata) {
        double temp = daylightApproximationColorTemperature(pixel);

        if (temp < warmThreshold1) warmCount++;
        else if (temp >= warmThreshold1 && temp <= coolThreshold1) neutralCount++;
        else coolCount++;
    }

    // Calculate a weighted warmth score (adjust weights as needed)
    double warmthScore = (warmCount * 1.0) + (neutralCount * 0.5) - (coolCount * 0.2);
    return warmthScore;
}

double filename_to_average_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);

    double totalTemperature = 0.0;
    for (const auto& pixel : rgbadata) {
        totalTemperature += daylightApproximationColorTemperature(pixel);
    }

    return rgbadata.empty() ? 0.0 : totalTemperature / rgbadata.size();
}
Refine the Calculation Method: Ensure both the manual calculation and your sorting code use identical steps for color temperature estimation, especially in chromaticity calculations.

Average Instead of Median: Try averaging the temperatures rather than taking the median, which might smooth out isolated color spikes.

Exclude Extreme Outliers: Consider ignoring outliers if they deviate significantly from the image's general hue.



----------------------------------------------------------------------------------------------------------


Your calculate_average_temperature function is straightforward and calculates the average color temperature by iterating over each pixel in the image. However, there are a few ways to improve its accuracy, efficiency, or robustness. Here are some suggestions:

1. Use Median Temperature Instead of Mean
Why: If an image has areas with extreme lighting (e.g., bright highlights or dark shadows), the average temperature might be skewed. The median temperature often provides a more stable result, as it’s less sensitive to outliers.
How: Instead of summing all temperatures and dividing by the count, store each pixel’s temperature in a vector, sort it, and return the median value.
cpp
Copy code
double calculate_median_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return 0;

    std::vector<double> temperatures;
    for (const auto& pixel : rgbadata) {
        temperatures.push_back(rgbToColorTemperature(pixel));
    }

    // Sort and find median
    std::sort(temperatures.begin(), temperatures.end());
    size_t size = temperatures.size();
    double median = (size % 2 == 0) ? (temperatures[size / 2 - 1] + temperatures[size / 2]) / 2.0 : temperatures[size / 2];
    return median;
}
2. Weighted Average Based on Brightness
Why: Certain areas in an image (such as bright or central regions) might contribute more to the perceived temperature. You could weight each pixel’s temperature based on its brightness (e.g., its Y component in XYZ color space or its grayscale value).
How: After converting RGB to XYZ, use the Y component as a weight when summing temperatures. Brighter pixels will contribute more to the final temperature.
cpp
Copy code
double calculate_weighted_average_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return 0;

    double weightedTemperature = 0.0;
    double totalWeight = 0.0;
    for (const auto& pixel : rgbadata) {
        double temperature = rgbToColorTemperature(pixel);
        // Convert pixel to XYZ to get brightness (Y component)
        double red = pixel.r / 255.0;
        double green = pixel.g / 255.0;
        double blue = pixel.b / 255.0;
        red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
        green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
        blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

        double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;

        weightedTemperature += temperature * Y;
        totalWeight += Y;
    }
    return (totalWeight == 0) ? 0.0 : weightedTemperature / totalWeight;
}
3. Block-Based Sampling for Large Images
Why: Processing every pixel in a high-resolution image can be computationally expensive. Sampling blocks of pixels (for example, taking every 5th pixel horizontally and vertically) can reduce computation time significantly without a large loss in accuracy.
How: Modify the loop to increment x and y by a fixed step (e.g., 5) when reading pixels, or process only every nth pixel in rgbadata.
cpp
Copy code
double calculate_sampled_average_temperature(const std::string& filename, int sampleStep = 5) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return 0;

    double totalTemperature = 0.0;
    int count = 0;
    for (int i = 0; i < rgbadata.size(); i += sampleStep) {
        totalTemperature += rgbToColorTemperature(rgbadata[i]);
        count++;
    }
    return (count == 0) ? 0.0 : totalTemperature / count;
}
4. Parallelized Temperature Calculation
Why: You can utilize multi-threading (as you are already using in threaded_sort) to speed up the temperature calculation.
How: Divide the pixels across multiple threads, where each thread computes the temperature for a subset of pixels. Finally, aggregate results from all threads.
cpp
Copy code
double calculate_parallel_average_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return 0;

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<double>> futures;
    size_t chunkSize = rgbadata.size() / numThreads;

    for (size_t i = 0; i < numThreads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&, i]() {
            double localTotal = 0.0;
            size_t start = i * chunkSize;
            size_t end = (i == numThreads - 1) ? rgbadata.size() : start + chunkSize;
            for (size_t j = start; j < end; ++j) {
                localTotal += rgbToColorTemperature(rgbadata[j]);
            }
            return localTotal;
        }));
    }

    double totalTemperature = 0.0;
    for (auto& future : futures) {
        totalTemperature += future.get();
    }
    return totalTemperature / rgbadata.size();
}
5. Refinement: Weighted Median Calculation
If accuracy is more important than speed, consider a weighted median instead of an average or simple median. This can be achieved by sorting based on brightness weight, which can improve perceived accuracy.
Summary
If you need more precision, try using the median temperature (calculate_median_temperature).
If you want to factor in brightness, use the weighted average method (calculate_weighted_average_temperature).
If performance is crucial, use block-based sampling or parallel calculation.
Each of these methods has trade-offs in terms of speed and accuracy. Let me know if you'd like further details on any specific approach!