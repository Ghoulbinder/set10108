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
    const char* filepath = "C:/set10108-cw/set10108/labs/cw1/dataset/shakespeare.txt";

    std::vector<char> file_data = read_file(filepath);
    if (file_data.empty())
        return -1;

    // Example word list
    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    std::cout << "CPU Results:" << endl;
    for (const char* word : words)
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