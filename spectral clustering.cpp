#include"Cluster.h"
#include<numeric>
#include<cmath>
using namespace SpectralClustring;
double PointSimilarity::linepearsonCorrelation(Line3D line1, Line3D line2)
{
    std::vector<double> vec1 = { line1.X2 - line1.X1, line1.Y2 - line1.Y1, line1.Z2 - line1.Z1 };
    std::vector<double> vec2 = { line2.X2 - line2.X1, line2.Y2 - line2.Y1, line2.Z2 - line2.Z1 };

    double sum1 = std::accumulate(vec1.begin(), vec1.end(), 0.0);
    double sum2 = std::accumulate(vec2.begin(), vec2.end(), 0.0);
    double mean1 = sum1 / 3;
    double mean2 = sum2 / 3;

    double numerator = 0.0;
    double denominator1 = 0.0;
    double denominator2 = 0.0;

    for (int i = 0; i < 3; ++i) {
        numerator += (vec1[i] - mean1) * (vec2[i] - mean2);
        denominator1 += std::pow(vec1[i] - mean1, 2);
        denominator2 += std::pow(vec2[i] - mean2, 2);
    }

    double denominator = std::sqrt(denominator1 * denominator2);

    if (denominator == 0) {
        return 0; // Avoid division by zero
    }

    return numerator / denominator;
}

double SpectralClustring::PointSimilarity::angleGap(Line3D line1, Line3D line2)
{
    double dot_product = (line1.X2 - line1.X1) * (line2.X2 - line2.X1)
        + (line1.Y2 - line1.Y1) * (line2.Y2 - line2.Y1)
        + (line1.Z2 - line1.Z1) * (line2.Z2 - line2.Z1);
    double magnitude1 = std::sqrt(std::pow(line1.X2 - line1.X1, 2)
        + std::pow(line1.Y2 - line1.Y1, 2)
        + std::pow(line1.Z2 - line1.Z1, 2));
    double magnitude2 = std::sqrt(std::pow(line2.X2 - line2.X1, 2)
        + std::pow(line2.Y2 - line2.Y1, 2)
        + std::pow(line2.Z2 - line2.Z1, 2));

    double cos_theta = dot_product / (magnitude1 * magnitude2);
    // Handle potential rounding errors
    if (cos_theta > 1.0) {
        cos_theta = 1.0;
    }
    else if (cos_theta < -1.0) {
        cos_theta = -1.0;
    }
    return std::acos(cos_theta);
}