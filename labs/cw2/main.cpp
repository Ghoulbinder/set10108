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

// Helper structure for RGBA pixels (a is safe to ignore for this coursework)
struct rgba_t
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Helper function to load RGB data from a file, as a contiguous array (row-major) of RGB triplets, where each of R,G,B is a uint8_t and ranges from 0 to 255
std::vector<rgba_t> load_rgb(const char * filename, int& width, int& height)
{
    int n;
    unsigned char *data = stbi_load(filename, &width, &height, &n, 4);
    const rgba_t* rgbadata = (rgba_t*)(data);
    std::vector<rgba_t> vec;
    vec.assign(rgbadata, rgbadata +width*height);
    stbi_image_free(data);
    return vec;
}

// Conversion to color temperature
double rgbToColorTemperature(rgba_t rgba) {
    // Normalize RGB values to [0, 1]
    double red = rgba.r / 255.0;
    double green = rgba.g / 255.0;
    double blue = rgba.b / 255.0;

    // Apply a gamma correction to RGB values (assumed gamma 2.2)
    red = (red > 0.04045) ? pow((red + 0.055) / 1.055, 2.4) : (red / 12.92);
    green = (green > 0.04045) ? pow((green + 0.055) / 1.055, 2.4) : (green / 12.92);
    blue = (blue > 0.04045) ? pow((blue + 0.055) / 1.055, 2.4) : (blue / 12.92);

    // Convert to XYZ color space
    double X = red * 0.4124 + green * 0.3576 + blue * 0.1805;
    double Y = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    double Z = red * 0.0193 + green * 0.1192 + blue * 0.9505;

    // Calculate chromaticity coordinates
    double x = X / (X + Y + Z);
    double y = Y / (X + Y + Z);

    // Approximate color temperature using McCamy's formula
    double n = (x - 0.3320) / (0.1858 - y);
    double CCT = 449.0 * n*n*n + 3525.0 * n*n + 6823.3 * n + 5520.33;

    return CCT;
}


// Calculate the median from an image filename
double filename_to_median(const std::string& filename)
{
    int width, height;
    auto rgbadata = load_rgb(filename.c_str(), width, height);
    std::vector<double> temperatures;
    std::transform(rgbadata.begin(), rgbadata.end(), std::back_inserter(temperatures), rgbToColorTemperature);
    std::sort(temperatures.begin(), temperatures.end());
    auto median = temperatures.size() % 2 ? 0.5 * (temperatures[temperatures.size() / 2 - 1] + temperatures[temperatures.size() / 2]) : temperatures[temperatures.size() / 2];
    return median;
}

// Static sort -- REFERENCE ONLY
void static_sort(std::vector<std::string>& filenames)
{
    std::sort(filenames.begin(), filenames.end(), [](const std::string& lhs, const std::string& rhs) {
        return filename_to_median(lhs) < filename_to_median(rhs);
    });
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
    const char* image_folder = "images/unsorted";
    if (!fs::is_directory(image_folder))
    {
        printf("Directory \"%s\" not found: please make sure it exists, and if it's a relative path, it's under your WORKING directory\n", image_folder);
        return -1;
    }
    std::vector<std::string> imageFilenames;
    for (auto& p : fs::directory_iterator(image_folder))
        imageFilenames.push_back(p.path().u8string());

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //  YOUR CODE HERE INSTEAD, TO ORDER THE IMAGES IN A MULTI-THREADED MANNER WITHOUT BLOCKING  //
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static_sort(imageFilenames);

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
    sf::Sprite sprite (texture);
    // Make sure the texture fits the screen
    sprite.setScale(SpriteScaleFromDimensions(texture.getSize(),gameWidth,gameHeight));

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
                view.setCenter(gameWidth/2.f, gameHeight/2.f);
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
