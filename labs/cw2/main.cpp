#include <SFML/Graphics.hpp>        // For graphics rendering
#include <cmath>                   // For mathematical operations
#include <ctime>                   // For time-based operations
#include <cstdlib>                 // For general utilities
#include <filesystem>              // For filesystem operations
#include <future>                  // For asynchronous operations and std::future
#include <vector>                  // For std::vector container
#include <algorithm>               // For sorting and algorithms
#include <iostream>                // For input/output streams
#include <mutex>                   // For std::mutex and std::lock_guard
#include <chrono>                  // For measuring time
#include <thread>                  // For std::thread
#include <queue>                   // For std::queue container
#include <functional>              // For std::function
#include <condition_variable>      // For std::condition_variable
#include <execution>               // For parallel STL (optional, depending on use)
#include <limits>                  // For numeric limits like std::numeric_limits
#include <numeric>                 // For std::iota

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

/*
-------------------------------TO DO LIST---------------------------------------------
        possible async for sorting can be done manually
        look into mccamys formual and judds formala for temp calculations
        current temp ranges can me improved




*/


namespace fs = std::filesystem;

// Helper structure for RGBA pixels (a is safe to ignore for this coursework)
struct rgba_t
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Helper function to load RGB data from a file, as a contiguous array (row-major) of RGB triplets, where each of R,G,B is a uint8_t and ranges from 0 to 255
std::vector<rgba_t> load_rgb(const char* filename, int& width, int& height)
{
    int n;
    unsigned char* data = stbi_load(filename, &width, &height, &n, 4);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    const rgba_t* rgbadata = (rgba_t*)(data);
    std::vector<rgba_t> vec;
    vec.assign(rgbadata, rgbadata + width * height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature
double rgbToColorTemperature(rgba_t rgba) {
    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply gamma correction to RGB values (assumed gamma 2.2)
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates, check for zero division
    double sum = X + Y + Z;
    if (sum == 0) {
        return 6500;  // Default temperature for neutral or missing data
    }
    double x = X / sum;
    double y = Y / sum;

    // Improved color temperature estimation using McCamy's formula
    double n = (x - 0.332) / (y - 0.1858);
    double CCT = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    // Empirical formula for color temperature estimation (Judd modified by CIE)
    //double n = (x - 0.3366) / (y - 0.1735);
    //double CCT = -949.86315 + 6253.80338 * exp(-n / 0.92159) + 28.70599 * exp(-n / 0.20039) +
    //    0.00004 * exp(-n / 0.07125);


    return CCT;
}

//double calculate_median_temperature(const std::string& filename) {
//    int width, height;
//    auto rgbadata = load_rgb(filename.c_str(), width, height);
//    if (rgbadata.empty()) return 0;
//
//    std::vector<double> temperatures;
//    for (const auto& pixel : rgbadata) {
//        temperatures.push_back(rgbToColorTemperature(pixel));
//    }
//
//    // Sort and find median
//    std::sort(temperatures.begin(), temperatures.end());
//    size_t size = temperatures.size();
//    double median = (size % 2 == 0) ? (temperatures[size / 2 - 1] + temperatures[size / 2]) / 2.0 : temperatures[size / 2];
//    return median;
//}

double calculate_median_temperature(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return 0;

    std::vector<double> temperatures(rgbadata.size());

    // Parallel processing with std::async
    const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;
    size_t chunkSize = rgbadata.size() / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = (t == numThreads - 1) ? rgbadata.size() : (t + 1) * chunkSize;

        futures.emplace_back(std::async(std::launch::async, [startIdx, endIdx, &rgbadata, &temperatures]() {
            for (size_t i = startIdx; i < endIdx; ++i) {
                temperatures[i] = rgbToColorTemperature(rgbadata[i]);
            }
            }));
    }

    // Wait for all threads to finish
    for (auto& fut : futures) {
        fut.get();
    }

    // Sort and calculate median
    std::nth_element(temperatures.begin(), temperatures.begin() + temperatures.size() / 2, temperatures.end());
    size_t mid = temperatures.size() / 2;
    if (temperatures.size() % 2 == 0) {
        double lower = *std::max_element(temperatures.begin(), temperatures.begin() + mid);
        double upper = temperatures[mid];
        return (lower + upper) / 2.0;
    }
    else {
        return temperatures[mid];
    }
}




// Multi-threaded sorting based on color temperature
//void threaded_sort(std::vector<std::string>& filenames) {
//    std::vector<std::future<double>> futures;
//
//    // Launch asynchronous tasks to calculate median temperatures for better stability
//    for (const auto& filename : filenames) {
//        futures.emplace_back(std::async(std::launch::async, calculate_median_temperature, filename));
//    }
//
//    // Collect temperatures and associate them with filenames
//    std::vector<std::pair<std::string, double>> temp_pairs;
//    for (size_t i = 0; i < filenames.size(); ++i) {
//        double temp = futures[i].get();
//        temp_pairs.emplace_back(filenames[i], temp);
//    }
//
//    // Sort filenames based on temperature
//    std::sort(temp_pairs.begin(), temp_pairs.end(), [](const auto& lhs, const auto& rhs) {
//        return lhs.second > rhs.second; //decending order
//        });
//
//    // Reorder filenames according to sorted temperatures
//    for (size_t i = 0; i < filenames.size(); ++i) {
//        filenames[i] = temp_pairs[i].first;
//    }
//}

void threaded_sort(std::vector<std::string>& filenames) {
    std::vector<std::pair<std::string, double>> temp_pairs(filenames.size());

    // Use parallel execution policy for temperature calculations
    std::transform(
        std::execution::par,
        filenames.begin(),
        filenames.end(),
        temp_pairs.begin(),
        [](const std::string& filename) -> std::pair<std::string, double> {
            return { filename, calculate_median_temperature(filename) };
        }
    );

    // Sort temp_pairs in descending order using parallel execution
    std::sort(
        std::execution::par,
        temp_pairs.begin(),
        temp_pairs.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        }
    );

    // Reorder filenames based on sorted order
    std::transform(
        temp_pairs.begin(),
        temp_pairs.end(),
        filenames.begin(),
        [](const auto& pair) {
            return pair.first;
        }
    );
}

std::pair<double, double> calculate_temperature_range(const std::string& filename) {
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    if (rgbadata.empty()) return { 0, 0 };

    double min_temp = std::numeric_limits<double>::max();
    double max_temp = std::numeric_limits<double>::lowest();

    for (const auto& pixel : rgbadata) {
        double temp = rgbToColorTemperature(pixel);
        if (temp < min_temp) min_temp = temp;
        if (temp > max_temp) max_temp = temp;
    }

    return { min_temp, max_temp };
}


sf::Vector2f SpriteScaleFromDimensions(const sf::Vector2u& textureSize, int screenWidth, int screenHeight)
{
    float scaleX = screenWidth / float(textureSize.x);
    float scaleY = screenHeight / float(textureSize.y);
    float scale = std::min(scaleX, scaleY);
    return { scale, scale };
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // example folder to load images
    const char* image_folder = "C:/set10108-cw/set10108/labs/cw2/images/unsorted";
    if (!fs::is_directory(image_folder))
    {
        printf("Directory \"%s\" not found: please make sure it exists, and if it's a relative path, it's under your WORKING directory\n", image_folder);
        return -1;
    }
    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder))
        imageFilenames.push_back(p.path().u8string());

    // Measure sorting time
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timer


    /////////////////////////////////////////////////////////////////////////////////////////////// 
    //  YOUR CODE HERE INSTEAD, TO ORDER THE IMAGES IN A MULTI-THREADED MANNER WITHOUT BLOCKING  //
    ///////////////////////////////////////////////////////////////////////////////////////////////
     // Sort images by temperature in a multi-threaded manner
    threaded_sort(imageFilenames);

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Sorting completed in " << duration.count() << " seconds.\n";

    //// Confirm sorting by printing
    //for (const auto& filename : imageFilenames) {
    //    double temp = calculate_median_temperature(filename);
    //    std::cout << "Image: " << filename << " - Temperature: " << temp << "K\n";
    //}
    for (const auto& filename : imageFilenames) {
        double assigned_temp = calculate_median_temperature(filename); // Assigned temp (median)
        std::cout << "Image: " << filename << " - Assigned Temperature: " << assigned_temp << "K\n";
    }


    // Define some constants
    const int gameWidth = 800;
    const int gameHeight = 600;

    int imageIndex = 0;

    // Create the window of the application
    sf::RenderWindow window(sf::VideoMode(gameWidth, gameHeight, 32), "Image Fever",
        sf::Style::Titlebar | sf::Style::Close);
    window.setVerticalSyncEnabled(true);

    // Load an image to begin with
    sf::Texture texture;
    if (!texture.loadFromFile(imageFilenames[imageIndex]))
        return EXIT_FAILURE;
    sf::Sprite sprite(texture);
    // Make sure the texture fits the screen
    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));

    sf::Clock clock;
    while (window.isOpen())
    {
        // Handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Window closed or escape key pressed: exit
            if ((event.type == sf::Event::Closed) ||
                ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape)))
            {
                window.close();
                break;
            }

            // Window size changed, adjust view appropriately
            if (event.type == sf::Event::Resized)
            {
                sf::View view;
                view.setSize(gameWidth, gameHeight);
                view.setCenter(gameWidth / 2.f, gameHeight / 2.f);
                window.setView(view);
            }

            // Arrow key handling!
            if (event.type == sf::Event::KeyPressed)
            {
                // adjust the image index
                if (event.key.code == sf::Keyboard::Key::Left)
                    imageIndex = (imageIndex + imageFilenames.size() - 1) % imageFilenames.size();
                else if (event.key.code == sf::Keyboard::Key::Right)
                    imageIndex = (imageIndex + 1) % imageFilenames.size();
                // get image filename
                const auto& imageFilename = imageFilenames[imageIndex];
                // set it as the window title 
                window.setTitle(imageFilename);
                // ... and load the appropriate texture, and put it in the sprite
                if (texture.loadFromFile(imageFilename))
                {
                    sprite = sf::Sprite(texture);
                    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(), gameWidth, gameHeight));
                }
            }
        }

        // Clear the window
        window.clear(sf::Color(0, 0, 0));
        // draw the sprite
        window.draw(sprite);
        // Display things on screen
        window.display();
    }

    return EXIT_SUCCESS;
}