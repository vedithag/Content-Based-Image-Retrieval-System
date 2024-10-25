/*
    Veditha Gudapati
    Content Based Image retrival
*/
//Face detector is been implemented in this part of the code
/*******************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>

using namespace cv;
using namespace std;

// Function to read feature vectors from a CSV file
unordered_map<string, vector<float>> readFeatureVectorsFromCSV(const string& csvPath) {
    ifstream file(csvPath);
    string line;
    unordered_map<string, vector<float>> featureVectors;

    // Read each line from the CSV file
    while (getline(file, line)) {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ','); // Assuming the first column is the filename

        // Read feature values into a vector
        vector<float> features(516); // Assuming each vector has 512 elements + 2 Sobel gradients
        for (int i = 0; i < 512; ++i) {
            ss >> features[i];
            if (ss.peek() == ',') ss.ignore();
        }

        // Store the feature vector in the unordered_map
        featureVectors[filename] = features;
    }

    return featureVectors;
}

// Function to compute cosine similarity between two vectors
float cosineSimilarity(const vector<float>& v1, const vector<float>& v2) {
    float dotProduct = 0.0;
    float normV1 = 0.0;
    float normV2 = 0.0;

    // Compute dot product and vector norms
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }

    // Compute cosine similarity
    normV1 = sqrt(normV1);
    normV2 = sqrt(normV2);

    if (normV1 == 0.0 || normV2 == 0.0) {
        return 0.0; // Prevent division by zero
    } else {
        return dotProduct / (normV1 * normV2);
    }
}

// Function to compute Sobel gradients of an image
vector<float> computeSobelGradients(const Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat sobelX, sobelY;
    Sobel(gray, sobelX, CV_64F, 1, 0);
    Sobel(gray, sobelY, CV_64F, 0, 1);

    // Compute mean Sobel gradients in the X and Y directions
    Scalar meanSobelX = mean(sobelX);
    Scalar meanSobelY = mean(sobelY);

    // Store the Sobel gradients in a vector
    vector<float> gradients = { static_cast<float>(meanSobelX[0]), static_cast<float>(meanSobelY[0]) };

    return gradients;
}

// Function for DNN-based image retrieval
vector<pair<double, string>> dnn_image_retrieval(const string& directory, const string& csvPath, const string& targetImagePath) {
    // Read feature vectors from the CSV file
    auto featureVectors = readFeatureVectorsFromCSV(csvPath);

    // Extract the filename from the target image path
    size_t pos = targetImagePath.find_last_of("/\\");
    string targetFilename = targetImagePath.substr(pos + 1);

    // Check if features for the target image are available
    if (featureVectors.find(targetFilename) == featureVectors.end()) {
        cerr << "Features for target image not found in CSV." << endl;
        return {};
    }

    // Get feature vector and Sobel gradients for the target image
    vector<float> targetFeatures = featureVectors[targetFilename];
    Mat targetImage = imread(targetImagePath);
    vector<float> targetGradients = computeSobelGradients(targetImage);

    // Container for storing image similarities
    vector<pair<double, string>> similarities;

    // Iterate through each image and compute similarities
    for (const auto& pair : featureVectors) {
        if (pair.first == targetFilename) continue; // Skip the target image

        // Compute cosine similarity based on DNN features
        float similarity = cosineSimilarity(targetFeatures, pair.second);

        // Read the image and compute Sobel gradients
        Mat image = imread(directory + "/" + pair.first);
        vector<float> gradients = computeSobelGradients(image);

        // Incorporate Sobel gradients into similarity computation
        float sobelSimilarity = 1 - (abs(targetGradients[0] - gradients[0]) + abs(targetGradients[1] - gradients[1])) / 255.0;

        // Combine DNN cosine similarity and Sobel similarity
        float combinedSimilarity = 0.5 * similarity + 0.5 * sobelSimilarity;

        // Store the combined similarity and filename in the container
        similarities.emplace_back(static_cast<double>(combinedSimilarity), pair.first);
    }

    // Sort similarities in descending order
    sort(similarities.begin(), similarities.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first > b.first;
    });

    return similarities;
}

// Main function
int main() {
    // Paths and filenames
    const string DIRECTORY_PATH = "/home/veditha/Desktop/prcv/olympus";
    const string CSV_PATH = "/home/veditha/Desktop/prcv/ResNet18_olym.csv";
    const string TARGET_IMAGE_PATH = "/home/veditha/Desktop/prcv/olympus/pic.0088.jpg";

    // DNN-based image retrieval
    auto results = dnn_image_retrieval(DIRECTORY_PATH, CSV_PATH, TARGET_IMAGE_PATH);

    // Sort results by similarity in descending order
    sort(results.begin(), results.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first > b.first;
    });

    // Display most similar pictures
    cout << "\nAll most similar pictures:" << endl;
    for (const auto& result : results) {
        cout << result.second << " (Similarity: " << result.first << ")" << endl;
        Mat image = imread(DIRECTORY_PATH + "/" + result.second);
        imshow("Most Similar", image);
        waitKey(0); // Wait for a key press for each image
    }

    return 0;
}
