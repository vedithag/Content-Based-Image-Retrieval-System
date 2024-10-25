/*
  Veditha Gudapati
  Content Based Image retrival
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>

using namespace cv;
using namespace std;

const string DIRECTORY_PATH = "/home/veditha/PRCV/Project_2(M1)/olympus";
const string CSV_PATH = "/home/veditha/PRCV/Project_2(M1)/ResNet18_olym.csv";
const string TARGET_IMAGE_PATH = "/home/veditha/PRCV/Project_2(M1)/olympus/pic.0535.jpg";

/*Task 1: Baseline feature Matching*/

// Function to compute baseline features
Mat computeBaselineFeatures(const Mat& image) {
    // Take the 7x7 square in the middle of the image
    Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    Mat feature = image(roi).clone().reshape(1, 1);
    return feature;
}

/*Task 2: Histogram feature Matching*/

// Function to compute histogram features
Mat computeHistogramFeatures(const Mat& image) {
    // Convert image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    
    // Calculate histogram for the H and S channels
    vector<Mat> channels;
    split(hsvImage, channels);
    int h_bins = 8; // Reduced to 8 bins
    int s_bins = 8; // Reduced to 8 bins
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int histChannels[] = { 0, 1 }; // Renamed to avoid conflict
    Mat hist;
    calcHist(&hsvImage, 1, histChannels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    
    return hist.reshape(1, 1);
}

/*Task 3: Multi-Histogram feature Matching*/

// Function to compute histogram features for the whole image
Mat computeWholeImageHistogram(const Mat& image) {
    // Convert image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Calculate histogram for the H and S channels
    vector<Mat> channels;
    split(hsvImage, channels);
    int h_bins = 16;
    int s_bins = 16;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int histChannels[] = { 0, 1 }; 
    Mat hist;
    calcHist(&hsvImage, 1, histChannels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

// Function to compute histogram features for the center region of the image
Mat computeCenterRegionHistogram(const Mat& image) {
    // Take the 7x7 square in the middle of the image
    Rect roi(image.cols / 2 - 3, image.rows / 2 - 3, 7, 7);
    Mat roiImage = image(roi);

    // Convert region to HSV color space
    Mat hsvROI;
    cvtColor(roiImage, hsvROI, COLOR_BGR2HSV);

    // Calculate histogram for the H and S channels
    vector<Mat> channels;
    split(hsvROI, channels);
    int h_bins = 16;
    int s_bins = 16;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int histChannels[] = { 0, 1 }; // Renamed to avoid conflict
    Mat hist;
    calcHist(&hsvROI, 1, histChannels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

/*Task 4: Texure and colour feature Matching*/

// Function to compute histogram features for the whole image texture using Sobel operator
Mat computeTextureFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Compute gradients using Sobel operator
    Mat gradX, gradY;
    Sobel(grayImage, gradX, CV_32F, 1, 0);
    Sobel(grayImage, gradY, CV_32F, 0, 1);

    // Compute gradient magnitude and angle
    Mat magnitude, angle;
    cartToPolar(gradX, gradY, magnitude, angle, true);

    // Calculate histogram for gradient magnitudes
    int histSize = 16;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist(&magnitude, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

    // Custom distance metric for multi-histogram matching
    double customDistance(const Mat& hist1, const Mat& hist2) {
    // Calculate histogram intersection
    double intersection = compareHist(hist1, hist2, HISTCMP_INTERSECT);

    // Compute weighted average distance
    // Weighted averaging can be adjusted as needed
    double weightedDistance = (0.6 * (1 - intersection)) + (0.4 * intersection);

    return weightedDistance;
}

/*Extensions*/

/*Histograms of Laws filter responses*/

// Function to compute texture features using Laws' masks
Mat computeLawsTextureFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Define Laws' masks
    float L5[] = {1,  4,  6,  4,  1};
    float E5[] = {-1, -2,  0,  2,  1};
    float S5[] = {-1,  0,  2,  0, -1};
    float W5[] = {-1,  2,  0, -2,  1};
    float R5[] = {1, -4,  6, -4,  1};

// Compute convolutions using Laws' masks
Mat L5E5, E5L5, L5L5, E5S5, S5E5, E5E5, S5S5, L5S5, S5L5;

// Convert Laws' masks to Mat objects
Mat L5Mat(1, 5, CV_32F, L5);
Mat E5Mat(1, 5, CV_32F, E5);
Mat S5Mat(1, 5, CV_32F, S5);

filter2D(grayImage, L5E5, -1, L5Mat * E5Mat.t());
filter2D(grayImage, E5L5, -1, E5Mat * L5Mat.t());
filter2D(grayImage, L5L5, -1, L5Mat * L5Mat.t());
filter2D(grayImage, E5S5, -1, E5Mat * S5Mat.t());
filter2D(grayImage, S5E5, -1, S5Mat * E5Mat.t());
filter2D(grayImage, E5E5, -1, E5Mat * E5Mat.t());
filter2D(grayImage, S5S5, -1, S5Mat * S5Mat.t());
filter2D(grayImage, L5S5, -1, L5Mat * S5Mat.t());
filter2D(grayImage, S5L5, -1, S5Mat * L5Mat.t());


    // Compute energy values for each texture filter
    double energyL5E5 = norm(L5E5, NORM_L2);
    double energyE5L5 = norm(E5L5, NORM_L2);
    double energyL5L5 = norm(L5L5, NORM_L2);
    double energyE5S5 = norm(E5S5, NORM_L2);
    double energyS5E5 = norm(S5E5, NORM_L2);
    double energyE5E5 = norm(E5E5, NORM_L2);
    double energyS5S5 = norm(S5S5, NORM_L2);
    double energyL5S5 = norm(L5S5, NORM_L2);
    double energyS5L5 = norm(S5L5, NORM_L2);

    // Construct feature vector
    Mat features = (Mat_<float>(1, 9) << energyL5E5, energyE5L5, energyL5L5, energyE5S5, energyS5E5, energyE5E5, energyS5S5, energyL5S5, energyS5L5);

    return features;
}

/*Histograms Fourier Transform filter responses*/

// Function to compute Fourier Transform-based features
Mat computeFourierTextureFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Convert the grayscale image to float32
    grayImage.convertTo(grayImage, CV_32F);

    // Compute 2D Fourier Transform
    Mat fourierImage;
    dft(grayImage, fourierImage, DFT_COMPLEX_OUTPUT);

    // Split the complex output into real and imaginary parts
    vector<Mat> planes;
    split(fourierImage, planes);

    // Compute magnitude spectrum
    Mat magSpectrum;
    magnitude(planes[0], planes[1], magSpectrum);

    // Resize the magnitude spectrum to a 16x16 image
    Mat resizedSpectrum;
    resize(magSpectrum, resizedSpectrum, Size(16, 16));

    // Normalize the resized spectrum
    normalize(resizedSpectrum, resizedSpectrum, 0, 1, NORM_MINMAX);

    return resizedSpectrum.reshape(1, 1);
}

/*Histograms of Gabor filter responses*/

// Function to compute Histograms of Gabor filter responses
Mat computeGaborHistogramFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Define Gabor filter parameters
    int kernelSize = 7; // Adjust the kernel size as needed
    double sigma = 2.0;
    double theta = 0.0;
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = 0.0;

    // Initialize Gabor filter bank
    vector<Mat> gaborFilterBank;

    // Generate Gabor filter responses for different orientations
    for (int i = 0; i < 8; ++i) {
        theta = i * CV_PI / 8.0;
        Mat gaborKernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, theta, lambda, gamma, psi);
        gaborFilterBank.push_back(gaborKernel);
    }

    // Compute Gabor filter responses for the image
    vector<Mat> gaborResponses;
    for (const auto& kernel : gaborFilterBank) {
        Mat response;
        filter2D(grayImage, response, CV_32F, kernel);
        gaborResponses.push_back(response);
    }

    // Concatenate Gabor filter responses into a feature vector
    Mat gaborFeatureVector;
    hconcat(gaborResponses, gaborFeatureVector);

    // Calculate histogram for the Gabor filter responses
    int histSize = 16; // Adjust the number of bins as needed
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist(&gaborFeatureVector, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

/*Task 5: Deep Network Embeddings*/

float L2Distance(const vector<float>& v1, const vector<float>& v2) {
    float sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
}

unordered_map<string, vector<float>> readFeatureVectorsFromCSV(const string& csvPath) {
    ifstream file(csvPath);
    string line;
    unordered_map<string, vector<float>> featureVectors;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ','); // Assuming the first column is the filename

        vector<float> features(512); // Assuming each vector has 512 elements
        for (int i = 0; i < 512; ++i) {
            ss >> features[i];
            if (ss.peek() == ',') ss.ignore();
        }

        featureVectors[filename] = features;
    }

    return featureVectors;
}

// Function to perform feature matching
void performFeatureMatching(const Mat& targetDescriptors, const vector<Mat>& referenceDescriptors, vector<DMatch>& matches) {
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->clear(); // Clear previous settings

    // Match descriptors
    matcher->add(referenceDescriptors);
    matcher->match(targetDescriptors, matches);
}

// Function to display matched images
void displayMatches(const Mat& targetImage, const Mat& referenceImage, const vector<KeyPoint>& targetKeypoints,
                    const vector<KeyPoint>& referenceKeypoints, const vector<DMatch>& matches) {
    Mat imgMatches;
    drawMatches(targetImage, targetKeypoints, referenceImage, referenceKeypoints, matches, imgMatches);

    namedWindow("Matches", WINDOW_NORMAL);
    imshow("Matches", imgMatches);
    waitKey(0);
}

/*Task 5: DNN*/

vector<pair<double, string>> dnn_image_retrieval(const string& directory, const string& csvPath, const string& targetImagePath) {
    auto featureVectors = readFeatureVectorsFromCSV(csvPath);

    size_t pos = targetImagePath.find_last_of("/\\");
    string targetFilename = targetImagePath.substr(pos + 1);
    if (featureVectors.find(targetFilename) == featureVectors.end()) {
        cerr << "Features for target image not found in CSV." << endl;
        return {};
    }
    vector<float> targetFeatures = featureVectors[targetFilename];

    vector<pair<double, string>> distances;
    for (const auto& pair : featureVectors) {
        if (pair.first == targetFilename) continue;
        float distance = L2Distance(targetFeatures, pair.second);
        distances.emplace_back(static_cast<double>(distance), pair.first);
    }

    sort(distances.begin(), distances.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first < b.first;
    });

    return distances;
}

void displayImage(const string& imagePath) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Could not open or find the image: " << imagePath << endl;
        return;
    }

    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", image);
    waitKey(0);
}

/*Task 7: Custom Design*/
/*has been implemented in separate file called task7.cpp file along with extentions*/


int main() {
    // Path to the target image
    string targetImagePath = TARGET_IMAGE_PATH;

    // Path to the directory containing the images for comparison
    string directory = DIRECTORY_PATH;

    bool runORB = false;

    // Read target image
    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty()) {
        cout << "Error: Unable to read target image " << targetImagePath << endl;
        return -1;
    }

    // Compute features for target image
    Mat targetBaselineFeatures = computeBaselineFeatures(targetImage);
    Mat targetHistogramFeatures = computeHistogramFeatures(targetImage);
    Mat targetWholeImageHist = computeWholeImageHistogram(targetImage);
    Mat targetCenterRegionHist = computeCenterRegionHistogram(targetImage);
    Mat targetTextureFeatures = computeTextureFeatures(targetImage);
    Mat targetLawsTextureFeatures = computeLawsTextureFeatures(targetImage);
    Mat targetFourierTextureFeatures = computeFourierTextureFeatures(targetImage);
    Mat targetGaborHistogramFeatures = computeGaborHistogramFeatures(targetImage);

    // Read images from directory
    vector<string> filenames;
    glob(directory + "/*.jpg", filenames);

    // Event loop
    while (true) {
        // Display target image
        imshow("Target Image", targetImage);

        // Check for key press
        int key = waitKey(0);
        if (key == 'b') { // 'b' key for baseline matching
            // Calculate distances for baseline features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageBaselineFeatures = computeBaselineFeatures(image);

                // Compute distance
                double distance = norm(targetBaselineFeatures, imageBaselineFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N baseline matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " baseline matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N baseline matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Baseline Match " + to_string(i + 1), image); // Start from 1
            }
        } else if (key == 'h') { // 'h' key for histogram matching
            // Calculate distances for histogram features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageHistogramFeatures = computeHistogramFeatures(image);

                // Compute distance
                double distance = norm(targetHistogramFeatures, imageHistogramFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N histogram matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " histogram matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N histogram matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Histogram Match " + to_string(i + 1), image); // Start from 1
            }

        runORB = false; // Reset the flag after processing ORB features

        } else if (key == 'g') { // 'g' key for Gabor histogram matching
            // Calculate distances for Gabor histogram features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageGaborHistogramFeatures = computeGaborHistogramFeatures(image);

                // Compute distance
                double distance = norm(targetGaborHistogramFeatures, imageGaborHistogramFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N Gabor histogram matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " Gabor histogram matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N Gabor histogram matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Gabor Histogram Match " + to_string(i + 1), image); // Start from 1
            }

            } else if (key == 'l') { // 'l' key for Laws' texture matching
            // Calculate distances for Laws' texture features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageLawsTextureFeatures = computeLawsTextureFeatures(image);

                // Compute distance
                double distance = norm(targetLawsTextureFeatures, imageLawsTextureFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N Laws' texture matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " Laws' texture matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N Laws' texture matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Laws' Texture Match " + to_string(i + 1), image); // Start from 1
            }

        } else if (key == 'x') { // 'x' key for Fourier texture matching
            // Calculate distances for Fourier texture features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageFourierTextureFeatures = computeFourierTextureFeatures(image);

                // Compute distance
                double distance = norm(targetFourierTextureFeatures, imageFourierTextureFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N Fourier texture matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " Fourier texture matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N Fourier texture matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Fourier Texture Match " + to_string(i + 1), image); // Start from 1
            }



        } else if (key == 'f') { // 'f' key for multi-histogram matching
            // Calculate distances for multi-histogram features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageWholeImageHist = computeWholeImageHistogram(image);
                Mat imageCenterRegionHist = computeCenterRegionHistogram(image);

                // Compute distance using custom distance metric
                double distance = customDistance(targetWholeImageHist, imageWholeImageHist) +
                                  customDistance(targetCenterRegionHist, imageCenterRegionHist);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N multi-histogram matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " multi-histogram matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N multi-histogram matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Multi-Histogram Match " + to_string(i + 1), image); // Start from 1
            }
        } else if (key == 't') { // 't' key for texture matching
            // Calculate distances for texture features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == targetImagePath) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    cout << "Error: Unable to read image " << filename << endl;
                    continue;
                }

                // Compute features
                Mat imageTextureFeatures = computeTextureFeatures(image);

                // Compute distance
                double distance = norm(targetTextureFeatures, imageTextureFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N texture matches
            int N = 3; // Change this value as needed
            cout << "Top " << N << " texture matches for the target image " << targetImagePath << ":" << endl;
            for (int i = 0; i < N && i < distances.size(); ++i) {
                cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N texture matches
            for (int i = 0; i < N && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Texture Match " + to_string(i + 1), image); // Start from 1
            }
        } else if (key == 'c') { // 'c' key for DNN-based image retrieval
            // DNN image retrieval
            auto results = dnn_image_retrieval(DIRECTORY_PATH, CSV_PATH, TARGET_IMAGE_PATH);

            // Display top 3 matches
            cout << "Top 3 matches:" << endl;
            int count = 0;
            for (const auto& result : results) {
                if (count >= 3) break;
                cout << result.second << " (Distance: " << result.first << ")" << endl;
                Mat image = imread(DIRECTORY_PATH + "/" + result.second);
                imshow("DNN Match " + to_string(count + 1), image); // Start from 1
                count++;
            }
        } else if (key == 27) { // ESC key to exit
            break; // Exit the program
        }
    }

    return 0;
}

