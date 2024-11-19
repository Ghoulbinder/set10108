// This is a chopped Pong example from SFML examples

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Graphics.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include <vector>
#include <future>  // For std::async and std::future
#include <mutex>   // For std::mutex
#include <chrono>    // For performance measurement
#include <execution>  // Include for parallel execution policies
#include <map>




/*
=====================================
            COURSEWORK TO-DO LIST
=====================================

1. Setup and Initial Implementation:
    - Initialize SFML and STB libraries.
    - Create a main window using SFML for displaying images.
    - Define file path for loading images.

    - Load and display images:
        - Create a function to load images from a specified folder.
        - Loop through the folder to collect image file names into a list.
        - Display the first image in the main window.

2. Implement Image Sorting by Color Temperature:
    - Define color temperature calculation:
        - Create a function to calculate color temperature for an image.
        - Load RGB data, convert RGB to color temperature using McCamy’s formula.
        - Return color temperature.

    - Sort images based on color temperature:
        - Create a function to sort images by temperature from warmest to coolest.
        - Sort images in ascending order to display them sequentially.

3. Integrate Multithreading for Sorting:
    - Move sorting to a separate thread:
        - Define a threaded sorting function using `std::async` or `std::thread`.
        - Lock the images list to avoid data races.
        - Sort images by color temperature in a background thread.
        - Use a `std::mutex` to ensure safe access to the images list.

    - Manage thread completion:
        - In the main program loop, check if sorting is complete.
        - Display images in sorted order once sorting is done.
        - Unlock the images list after sorting completes.

4. Implement Event Handling for User Interaction:
    - Handle left/right arrow key presses:
        - Change the currently displayed image based on key press (left or right).
        - Update the window title with the current image file name.

    - Handle window resizing:
        - Adjust image scale to fit the new window size as it resizes.

5. Measure Performance for Report:
    - Add timing functions around the sorting process to measure duration:
        - Use `std::chrono` to record time before and after sorting.
        - Store and display timing results for performance analysis.

    - Experiment with different sorting or multithreading techniques, if applicable.
    - Record any improvements in sorting time for reporting.

6. Write Report:
    - Analysis Section:
        - Explain the image sorting process.
        - Describe concurrency techniques used, and the rationale behind them.

    - Methodology and Results:
        - Present timing and performance metrics from the multithreaded sorting.
        - Include graphs or tables to compare single-threaded vs. multi-threaded performance, if possible.

7. Refactor Code (Optional, After Testing):
    - Move image loading and sorting functions into dedicated files:
        - Create "ImageLoader.h/.cpp" and "ImageSorter.h/.cpp" for specific functionality.
        - Create "Utils.h/.cpp" for helper functions like color temperature calculations.
    - Update `main.cpp` to use these modular components for a clean, organized structure.

=====================================
*/

namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION

// Helper structure for RGBA pixels (a is safe to ignore for this coursework)
struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// Mutex for protecting shared access to image filenames
std::mutex filenames_mutex;

// Helper function to load RGB data from a file, as a contiguous array of RGB triplets
std::vector<rgba_t> load_rgb(const char* filename, int& width, int& height) {
    int n;
    unsigned char* data = stbi_load(filename, &width, &height, &n, 4);
    const rgba_t* rgbadata = (rgba_t*)(data);
    std::vector<rgba_t> vec;
    vec.assign(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature
double daylightApproximationColorTemperature(rgba_t rgba) {
    // Step 1: Normalize RGB values to the range [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Step 2: Apply gamma correction to RGB values
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Step 3: Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Step 4: Calculate chromaticity coordinates x and y
    double x = X / (X + Y + Z);
    double y = Y / (X + Y + Z);

    // Step 5: Approximate CCT using Daylight Chromaticity Approximation (CIE Method)
    double n = (x - 0.332) / (y - 0.1858);
    double CCT = 437 * n * n * n + 3601 * n * n + 6861 * n + 5517;

    return CCT;
}

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

/*double combined_temperature_score(const std::string& filename) {
    // Calculate both metrics
    double avgTemperature = filename_to_average_temperature(filename);
    double dominantRatio = filename_to_dominant_temperature(filename);

    // Weighting factors (adjust these as necessary for best results)
    const double avgWeight = 0.6;
    const double dominantWeight = 0.4;

    // Return a combined score
    return (avgWeight * avgTemperature) + (dominantWeight * (dominantRatio * 10000)); // Scale ratio to balance with avg
}*/

/*// Static sort (descending order by combined temperature score)
void static_sort(std::vector<std::string>& filenames) {
    std::sort(filenames.begin(), filenames.end(), [](const std::string& lhs, const std::string& rhs) {
        return combined_temperature_score(lhs) > combined_temperature_score(rhs);  // Sort by combined score
        });
}*/

// Use histogram - based warmth score sorting function
void threaded_sort(std::vector<std::string>&filenames) {
    auto start = std::chrono::high_resolution_clock::now();

    {
        std::lock_guard<std::mutex> lock(filenames_mutex);
        std::sort(std::execution::par, filenames.begin(), filenames.end(), [](const std::string& lhs, const std::string& rhs) {
            return calculateHistogramWarmthScore(lhs) > calculateHistogramWarmthScore(rhs);
            });
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Histogram-based sorting took " << elapsed.count() << " seconds\n";
}
// Parallelized sorting function with timing measurement (descending order by combined temperature score)
/*void threaded_sort(std::vector<std::string>& filenames) {
    auto start = std::chrono::high_resolution_clock::now();  // Start timing

    {
        std::lock_guard<std::mutex> lock(filenames_mutex);
        // Parallel sort using std::execution::par for parallel execution
        std::sort(std::execution::par, filenames.begin(), filenames.end(), [](const std::string& lhs, const std::string& rhs) {
            return combined_temperature_score(lhs) > combined_temperature_score(rhs);
            });
    }

    auto end = std::chrono::high_resolution_clock::now();  // End timing
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sorting took " << elapsed.count() << " seconds\n";
}*/

// Calculate sprite scale for the image to fit within window
sf::Vector2f SpriteScaleFromDimensions(const sf::Vector2u& textureSize, int screenWidth, int screenHeight) {
    float scaleX = screenWidth / float(textureSize.x);
    float scaleY = screenHeight / float(textureSize.y);
    float scale = std::min(scaleX, scaleY);
    return { scale, scale };
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Folder to load images
    const char* image_folder = "C:/set10108-cw/set10108/labs/cw2/build/images/unsorted";
    if (!fs::is_directory(image_folder)) {
        printf("Directory \"%s\" not found\n", image_folder);
        return -1;
    }

    // Load image filenames
    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder))
        imageFilenames.push_back(p.path().u8string());

    // Asynchronously start sorting images by color temperature
    auto sort_future = std::async(std::launch::async, threaded_sort, std::ref(imageFilenames));

    // Define window settings
    const int gameWidth = 800;
    const int gameHeight = 600;
    int imageIndex = 0;
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Image Fever", sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    // Load the first image
    sf::Texture texture;
    if (!texture.loadFromFile(imageFilenames[imageIndex]))
        return EXIT_FAILURE;
    sf::Sprite sprite(texture);
    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));

    bool sortingComplete = false;  // Flag to track sorting completion

    while (window.isOpen()) {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event)) {
            if ((event.type == sf::Event::Closed) ||
                ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape))) {
                window.close();
                break;
            }

            // Window resize event
            if (event.type == sf::Event::Resized) {
                sf::View view;
                view.setSize(gameWidth, gameHeight);
                view.setCenter(gameWidth / 2.f, gameHeight / 2.f);
                window.setView(view);
            }

            // Handle arrow key to change images
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Left)
                    imageIndex = (imageIndex + imageFilenames.size() - 1) % imageFilenames.size();
                else if (event.key.code == sf::Keyboard::Right)
                    imageIndex = (imageIndex + 1) % imageFilenames.size();

                // Update the title with the current image filename
                const auto& imageFilename = imageFilenames[imageIndex];
                window.setTitle(imageFilename);

                // Load the new image texture
                if (texture.loadFromFile(imageFilename)) {
                    sprite = sf::Sprite(texture);
                    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                }
            }
        }

        // Check if sorting is complete
        if (!sortingComplete && sort_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            std::lock_guard<std::mutex> lock(filenames_mutex);
            sortingComplete = true;  // Set flag to avoid re-checking
            imageIndex = 0;  // Reset to the first image in the sorted list

            // Load the first image from the sorted order
            if (texture.loadFromFile(imageFilenames[imageIndex])) {
                sprite = sf::Sprite(texture);
                sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                window.setTitle(imageFilenames[imageIndex]);  // Update window title
            }
        }

        // Render the current frame
        window.clear(sf::Color(0, 0, 0));
        window.draw(sprite);
        window.display();
    }

    return EXIT_SUCCESS;
}