#include"Cluster.h"
#include<numeric>
#include<cmath>

#define M_PI 3.14159265358979323846
using namespace SpectralClustring;


//计算了这两个坐标之间的差值作为方向向量，最后计算了这两个方向向量之间的相关性。
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

//通过计算两个向量的点积和模长，然后使用反余弦函数计算它们之间的夹角。
// 需要注意的是，计算夹角时可能会有舍入误差，因此在取反余弦函数之前，应该确保夹角的余弦值在 [-1, 1] 的范围内。
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


LinearFeature SpectralClustring::PointSimilarity::linearFeature(Line3D line)
{
    LinearFeature feature;

    // Calculate midpoint coordinates
    feature.x_midpoint = (line.X1 + line.X2) / 2.0;
    feature.y_midpoint = (line.Y1 + line.Y2) / 2.0;
    feature.z_midpoint = (line.Z1 + line.Z2) / 2.0;

    // Calculate latitude and longitude
    double radius = std::sqrt(std::pow(feature.x_midpoint, 2) + std::pow(feature.y_midpoint, 2) + std::pow(feature.z_midpoint, 2));
    feature.latitude = std::asin(feature.z_midpoint / radius) * 180.0 / M_PI;
    feature.longitude = std::atan2(feature.y_midpoint, feature.x_midpoint) * 180.0 / M_PI;

    return feature;
}

double SpectralClustring::PointSimilarity::straightLineDistance(Line3D line1, Line3D line2)
{
    double x1 = line1.X1, y1 = line1.Y1, z1 = line1.Z1;
    double x2 = line1.X2, y2 = line1.Y2, z2 = line1.Z2;
    double x3 = line2.X1, y3 = line2.Y1, z3 = line2.Z1;
    double x4 = line2.X2, y4 = line2.Y2, z4 = line2.Z2;

    double a = (y4 - y3) * (z2 - z1) - (z4 - z3) * (y2 - y1);
    double b = (z4 - z3) * (x2 - x1) - (x4 - x3) * (z2 - z1);
    double c = (x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1);

    if (std::abs(c) < 1e-8) { // Lines are parallel
        return pointLineDistance(line1, line2); // Calculate distance to a point on one of the lines
    }

    double d = -(a * x3 + b * y3 + c * z3);

    // Calculate intersection point
    double t1 = -((a * x1 + b * y1 + c * z1 + d) / (a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1)));
    double x_inter = x1 + t1 * (x2 - x1);
    double y_inter = y1 + t1 * (y2 - y1);
    double z_inter = z1 + t1 * (z2 - z1);

    // Check if intersection point is within line segments
    if (t1 >= 0 && t1 <= 1 && std::min(x1, x2) <= x_inter && x_inter <= std::max(x1, x2)
        && std::min(y1, y2) <= y_inter && y_inter <= std::max(y1, y2)
        && std::min(z1, z2) <= z_inter && z_inter <= std::max(z1, z2)) {
        return 0.0; // Lines intersect, distance is 0
    }

    // If lines do not intersect, calculate minimum distance to any of the endpoints
    double dist1 = pointLineDistance(line1, { x3, y3, z3, x4, y4, z4 });
    double dist2 = pointLineDistance(line2, { x1, y1, z1, x2, y2, z2 });
    return std::min(dist1, dist2);
}


double SpectralClustring::PointSimilarity::pointLineDistance(Line3D line, Line3D point)
{
    double x1 = line.X1, y1 = line.Y1, z1 = line.Z1;
    double x2 = line.X2, y2 = line.Y2, z2 = line.Z2;
    double x3 = point.X1, y3 = point.Y1, z3 = point.Z1;
    double x4 = point.X2, y4 = point.Y2, z4 = point.Z2;

    double a = y2 - y1;
    double b = x1 - x2;
    double c = x2 * y1 - x1 * y2;

    double distance = std::abs(a * x3 + b * y3 + c) / std::sqrt(a * a + b * b);

    return distance;
}

double SpectralClustring::PointSimilarity::normalizedDistance(double lineDistance, MinMaxDistance distance)
{
    double minDistance = distance.minDistance;
    double maxDistance = distance.maxDistance;

    // 确保 lineDistance 在最小值和最大值之间
    lineDistance = std::max(minDistance, std::min(maxDistance, lineDistance));

    // 将 lineDistance 归一化到 [0, 1] 范围内
    double normalized = (lineDistance - minDistance) / (maxDistance - minDistance);
    return normalized;
}


MinMaxDistance SpectralClustring::PointSimilarity::minMaxDistance()
{
    double minDist = std::numeric_limits<double>::max();
    double maxDist = std::numeric_limits<double>::lowest();

    for (const auto& line : lines) {
        double dist = // Calculate distance of line using line.X1, line.Y1, etc.
            minDist = std::min(minDist, dist);
        maxDist = std::max(maxDist, dist);
    }

    return { minDist, maxDist };
}

double SpectralClustring::PointSimilarity::normalizedAngleGap(double angle)
{
    // 将角度限制在 [0, 2π) 范围内
    angle = fmod(angle, 2 * M_PI);
    if (angle < 0) {
        angle += 2 * M_PI;
    }
    // 归一化到 [0, 1] 范围内
    double normalized = angle / (2 * M_PI);
    return normalized;
}
double SpectralClustring::PointSimilarity::normalizedLinepearsonCorrelation(double linCorr)
{
    double normalized = (linCorr + 1.0) / 2.0; // 将 [-1, 1] 映射到 [0, 1]
    return normalized;
}

double SpectralClustring::PointSimilarity::assignmentSimilarity()
{
    double p1, p2, p3;

}

void SpectralClustring::PointSimilarity::Grapes()
{
    for (size_t i = 0; i < lines.size(); ++i) {
        serialNumberToIndexMap[lines[i].serialNumber] = i;
    }
}

// 获取对应行列元素
double PointSimilarity::getSimilarity(int row, int col) const
{
    // 检查行列是否有效
    if (row >= 0 && row < similarityMatrix.size() &&
        col >= 0 && col < similarityMatrix[row].size())
    {
        return similarityMatrix[row][col];
    }
    else
    {
        // 处理错误，这里简单地返回一个默认值
        return 0.0;
    }
}

