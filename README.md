# **Content-Based Image Retrieval (CBIR) Project**

## **Project Overview**

This project implements a Content-Based Image Retrieval (CBIR) system using various techniques such as baseline feature matching, histogram comparison, texture analysis, and deep learning embeddings to find similar images from a given image database. The system retrieves images based on their visual content, employing both traditional feature extraction methods and deep learning-based embeddings to compare and rank the similarity of images.

This project demonstrates expertise in image processing, computer vision, and machine learning, utilizing OpenCV and deep learning techniques to build a robust CBIR system. The system can extract and compare features based on color, texture, gradients, and neural network embeddings.

---

## **Core Technologies**
- **Programming Language:** C++
- **Library:** OpenCV
- **Deep Learning Model:** ResNet18 (Pre-trained on ImageNet)

---

## **Features Implemented**

1. **Baseline Matching:**
   - Uses regions of interest (ROI) in the image to extract a simple feature vector from the center of the image.
   - **Distance metric:** Euclidean distance (L2).

2. **Histogram Matching:**
   - HSV histograms are calculated for color information. Histograms are normalized and compared to find the best matches.
   - **Distance metric:** L2 distance between histograms.

3. **Multi-Histogram Matching:**
   - Combines histograms from the whole image and the central region for comparison, adding more detail to the feature extraction process.
   - **Custom distance metrics** are used for better matching accuracy.

4. **Texture and Gradient Matching:**
   - Texture features are extracted using Sobel gradients, which emphasize edges in the image. These are combined with color histograms for more accurate retrieval.
   - The system also uses **Laws texture filters**, **Gabor filters**, and **Fourier transforms** for advanced texture matching.

5. **DNN Embeddings:**
   - Feature vectors from a pre-trained **ResNet18** model are used to capture high-level semantic information from the images.
   - **Cosine similarity** and **L2 distance** are used for comparing the deep neural network embeddings.
   - The system efficiently retrieves semantically similar images using these learned features.

---

## **Advanced Techniques**
- **Custom Distance Metrics:** Combines color, texture, and gradient-based features using weighted combinations of distance scores.
- **Sobel Gradient Analysis:** Sobel filters are applied to extract edge-based texture features for image comparison.
- **Laws Texture Filters & Gabor Filters:** These filters are used to compute texture features from the image and match them with others in the dataset.
- **ResNet18 Embeddings:** Pre-trained ResNet18 features are used to extract semantic-level information for more meaningful image retrieval.

---

## **Dependencies**
- **C++ Compiler**: GCC or any C++11 compatible compiler.
- **OpenCV 4.x**: Required for image manipulation and feature extraction.
- **Pre-trained ResNet18 feature vectors**: Feature vectors in CSV format.

## **Key Technical Skills Demonstrated**
### **Computer Vision & Image Processing**
- Proficient in feature extraction techniques, including color histograms, texture filters, and edge-based features.
- Use of deep learning models for feature embedding and high-level image similarity.
- Expertise in OpenCV for image manipulation, processing, and comparison.

### **C++ Programming**
- Developed a complex, efficient image retrieval system using C++.
- Applied STL containers like `unordered_map` and `vector` for high-performance data handling.
- Modular, well-structured C++ code with emphasis on performance optimization.

### **Machine Learning & Deep Learning**
- Applied deep learning models (ResNet18) to extract semantic features from images.
- Used a combination of distance metrics like cosine similarity and L2 distance for high-dimensional vector comparison.
- Integrated traditional image processing methods with neural network-based embeddings for improved retrieval accuracy.
