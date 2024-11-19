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
