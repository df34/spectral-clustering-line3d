#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include<opencv2/opencv.hpp>
using namespace Eigen;

// Spectral clustering function
std::vector<int> spectralClustering(const std::vector<std::vector<double>>& similarityMatrix, int numClusters) {
    int n = similarityMatrix.size();
    Map<const MatrixXd> A(&similarityMatrix[0][0], n, n); // Convert to Eigen matrix

    // Compute degree matrix
    VectorXd D = A.rowwise().sum().array().sqrt().matrix().asDiagonal();

    // Compute Laplacian matrix
    MatrixXd L = D.inverse() * A * D.inverse();

    // Compute the eigenvectors of the Laplacian
    SelfAdjointEigenSolver<MatrixXd> eigensolver(L);
    MatrixXd eigenvectors = eigensolver.eigenvectors().real();

    // Use k-means clustering on the eigenvectors
    // (You can use your own k-means implementation or a library like OpenCV for this)
    // Here's a simple example using k-means from OpenCV
    std::vector<int> labels;
    cv::Mat points(eigenvectors.rows(), eigenvectors.cols(), CV_64F, eigenvectors.data());
    cv::Mat bestLabels, centers;
    cv::kmeans(points, numClusters, bestLabels, cv::TermCriteria(), 10, cv::KMEANS_PP_CENTERS, centers);

    // Convert the labels to std::vector<int>
    labels.assign(bestLabels.begin<int>(), bestLabels.end<int>());

    return labels;
}

int main() {
    // Example similarity matrix (replace with your own data)
    std::vector<std::vector<double>> similarityMatrix = {
        {1.0, 0.2, 0.1, 0.0},
        {0.2, 1.0, 0.3, 0.0},
        {0.1, 0.3, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    int numClusters = 2; // Number of clusters
    std::vector<int> labels = spectralClustering(similarityMatrix, numClusters);

    // Output the clustering result
    std::cout << "Clustering result:" << std::endl;
    for (int i = 0; i < labels.size(); ++i) {
        std::cout << "Point " << i << " belongs to cluster " << labels[i] << std::endl;
    }

    return 0;
}
