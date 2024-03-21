#include"Cluster.h"
#include<numeric>
#include<cmath>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include<Eigen/Eigenvalues>
#include<Eigen/src/Eigenvalues/EigenSolver.h>
#include<Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include<Spectra/MatOp/SparseSymMatProd.h>
#include<random>
#include<algorithm>
#include<opencv2/opencv.hpp>
//#include<matplotlibcpp.h>

#define M_PI 3.14159265358979323846
using namespace SpectralClustring;
using namespace Eigen;
using namespace Spectra;

typedef Eigen::SparseMatrix<double>SpMat;//声明一个doule类型的列主序稀疏矩阵
typedef Eigen::Triplet<double>T;

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
	double dist1 = pointLineDistance(line1, { x3, y3, z3, x4, y4, z4 ,line1.ID });
	double dist2 = pointLineDistance(line2, { x1, y1, z1, x2, y2, z2 ,line2.ID });
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

double SpectralClustring::PointSimilarity::assignmentSimilarity(Line3D line1, Line3D line2)
{
	double p1 = 0.4, p2 = 0.3, p3 = 0.3;
	double similarity = 0;
	double angle = angleGap(line1, line2);
	if (angle < M_PI / 12)
	{
		double lineDistance = straightLineDistance(line1, line2);
		MinMaxDistance minMaxDistances = minMaxDistance();
		double corr = linepearsonCorrelation(line1, line2);
		similarity = p1 * normalizedDistance(lineDistance, minMaxDistances) + p2 * normalizedAngleGap(angle) + p3 * normalizedLinepearsonCorrelation(corr);
	}
	else
	{
		similarity = 0;
	}
	return similarity;
}

void SpectralClustring::PointSimilarity::updateSimilarityMatrix(Line3D line1, Line3D line2)
{
	double similarity = assignmentSimilarity(line1, line2);

	// 将计算的相似度赋给对应直线的行列
	int id1 = line1.ID;
	int id2 = line2.ID;

	// 确保 similarityMatrix 具有足够的大小
	int newSize = std::max(id1, id2) + 1;
	if (similarityMatrix.size() < newSize)
	{
		similarityMatrix.resize(newSize, std::vector<double>(newSize, 0.0));
	}

	similarityMatrix[id1][id2] = similarity;
	similarityMatrix[id2][id1] = similarity;
}


void SpectralClustring::PointSimilarity::Grapes()
{
	for (size_t i = 0; i < lines.size(); ++i) {
		serialNumberToIndexMap[lines[i].ID] = i;
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

void SpectralClustring::LaplacianMatrix::calculateMetricMatrix(std::vector<std::vector<double>>& metricMatrix)
{
	metricMatrix.clear();
	metricMatrix.resize(similarityMatrix.size(), std::vector<double>(similarityMatrix.size(), 0.0));
	//计算对角线元素，对应节点的所有权值的累加值
	for (size_t i = 0; i < similarityMatrix.size(); ++i)
	{
		double sum = 0.0;
		for (size_t j = 0; j < similarityMatrix[i].size(); ++j)
		{
			sum += similarityMatrix[i][j];
		}
		metricMatrix[i][i] = sum;
	}
	for (size_t i = 0; i < metricMatrix.size(); ++i)
	{
		metricMatrix[i][i] = i; // 假设节点的ID就是索引值
	}
}

void SpectralClustring::LaplacianMatrix::calculateLaplacianMatrix()
{
	LaplacianMatrix.clear();
	LaplacianMatrix.resize(metricMatrix.size(), std::vector<double>(metricMatrix.size(), 0.0));

	// 计算拉普拉斯矩阵
	for (size_t i = 0; i < metricMatrix.size(); ++i)
	{
		for (size_t j = 0; j < metricMatrix[i].size(); ++j)
		{
			if (i == j)
			{
				// 对角线元素为节点的度数
				LaplacianMatrix[i][j] = metricMatrix[i][j];
			}
			else
			{
				// 非对角线元素为相似度或连接权重的负值
				LaplacianMatrix[i][j] = -metricMatrix[i][j];
			}
		}
	}

	// D - W
	for (size_t i = 0; i < metricMatrix.size(); ++i)
	{
		for (size_t j = 0; j < metricMatrix[i].size(); ++j)
		{
			LaplacianMatrix[i][j] = metricMatrix[i][j] - LaplacianMatrix[i][j];
		}
	}
}

void SpectralClustring::LaplacianMatrix::convertToSparseMatrix(const std::vector<std::vector<double>>& denseMatrix)
{
	int rows = denseMatrix.size();
	int cols = denseMatrix[0].size();

	// 创建一个稀疏矩阵
	SparseMatrix<double> sparseMatrix(rows, cols);

	// 遍历稠密矩阵，将非零元素添加到稀疏矩阵中
	std::vector<Triplet<double>> triplets;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if (denseMatrix[i][j] != 0.0)
			{
				triplets.push_back(Triplet<double>(i, j, denseMatrix[i][j]));
			}
		}
	}

	sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

	LaplacianMatrixSparse = sparseMatrix;
}

void SpectralClustring::LaplacianMatrix::calSparseMatrix()
{
	// 创建稀疏矩阵的三元组表示
	std::vector<Eigen::Triplet<double>> tripletList;

	// 遍历密集矩阵的元素，将非零元素添加到三元组列表中
	for (int i = 0; i < LaplacianMatrix.size(); ++i) {
		for (int j = 0; j < LaplacianMatrix[i].size(); ++j) {
			if (LaplacianMatrix[i][j] != 0.0) {
				tripletList.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrix[i][j]));
			}
		}
	}

	// 将三元组列表设置为稀疏矩阵的值
	LaplacianMatrixSparse.setFromTriplets(tripletList.begin(), tripletList.end());

	// 完成矩阵插入后进行最终化
	LaplacianMatrixSparse.finalize();
}

void SpectralClustring::LaplacianMatrix::computeEigen()
{
	/*EigenSolver<SpMat>es(LaplacianMatrixSparse);
	VectorXcd eigenvalues = es.eigenvalues();
	MatrixXcd eigenvectors = es.eigenvectors();*/
	SparseSymMatProd<double>op(LaplacianMatrixSparse);
	Spectra::SymEigsSolver< Spectra::SparseSymMatProd<double> > eigs(op, LaplacianMatrixSparse.rows(), 2 * LaplacianMatrixSparse.rows());
	eigs.init();
	int nconv = eigs.compute();
	// 获取特征值
	Eigen::VectorXd eigenvalues;
	if (eigs.info() == Spectra::CompInfo::Successful)
	{
		eigenvalues = eigs.eigenvalues();
	}
	// 获取特征向量
	Eigen::MatrixXd eigenvectors;
	if (eigs.info() == Spectra::CompInfo::Successful)
	{
		eigenvectors = eigs.eigenvectors();
	}
	// 将特征值和特征向量存储到一个键对的数组中

	for (int i = 0; i < eigenvalues.size(); ++i) {
		eigen_pairs.push_back(std::make_pair(eigenvalues[i], eigenvectors.col(i)));
	}

	// 打印特征值和特征向量
	for (const auto& pair : eigen_pairs)
	{
		std::cout << "Eigenvalue: " << pair.first << std::endl;
		std::cout << "Eigenvector: " << pair.second.transpose() << std::endl;
	}
}
void SpectralClustring::LaplacianMatrix::fileterEigenPairs()
{
	// 筛选特征值大小
	std::sort(eigen_pairs.begin(), eigen_pairs.end(),
		[](const std::pair<double, Eigen::VectorXd>& a, const std::pair<double, Eigen::VectorXd>& b) {
			return a.first < b.first;
		});

	// 找到第一个非零特征值的索引
	size_t firstNonZeroIndex = 0;
	while (firstNonZeroIndex < eigen_pairs.size() && eigen_pairs[firstNonZeroIndex].first == 0) {
		++firstNonZeroIndex;
	}

	// 保留0特征值和除零特征值以外最小的特征值及其对应的特征向量
	std::vector<std::pair<double, Eigen::VectorXd>> filteredEigenPairs;
	filteredEigenPairs.reserve(2);  // 保留0特征值和除零特征值以外最小的特征值及其对应的特征向量
	if (firstNonZeroIndex < eigen_pairs.size()) {
		filteredEigenPairs.push_back(eigen_pairs[firstNonZeroIndex]);
	}
	if (firstNonZeroIndex + 1 < eigen_pairs.size()) {
		filteredEigenPairs.push_back(eigen_pairs[firstNonZeroIndex + 1]);
	}

	eigen_pairs = filteredEigenPairs;
}

std::vector<int> SpectralClustring::LaplacianMatrix::kMeansClustering(const std::vector<Eigen::VectorXd>& data, int k, int maxIterations)
{
	// 初始化聚类中心
	std::vector<Eigen::VectorXd> centroids(k);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, data.size() - 1);
	for (int i = 0; i < k; ++i) {
		centroids[i] = data[dist(gen)];
	}

	// 开始迭代
	std::vector<int> assignments(data.size());
	for (int iter = 0; iter < maxIterations; ++iter) {
		// 分配样本到最近的聚类中心
		for (size_t i = 0; i < data.size(); ++i) {
			double minDistance = (data[i] - centroids[0]).norm();
			assignments[i] = 0;
			for (int j = 1; j < k; ++j) {
				double distance = (data[i] - centroids[j]).norm();
				if (distance < minDistance) {
					minDistance = distance;
					assignments[i] = j;
				}
			}
		}

		// 更新聚类中心
		std::vector<int> counts(k, 0);
		std::vector<Eigen::VectorXd> sums(k, Eigen::VectorXd::Zero(data[0].size()));
		for (size_t i = 0; i < data.size(); ++i) {
			int cluster = assignments[i];
			sums[cluster] += data[i];
			counts[cluster]++;
		}
		for (int i = 0; i < k; ++i) {
			if (counts[i] > 0) {
				centroids[i] = sums[i] / counts[i];
			}
		}
	}

	return assignments;
}

int SpectralClustring::LaplacianMatrix::selectKByElbowMethod(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations)
{
	std::vector<double> losses(maxK);

	for (int k = 1; k <= maxK; ++k) {
		std::vector<int> assignments = kMeansClustering(data, k, maxIterations);

		// 计算聚类损失（平均簇内平方和）
		double loss = 0.0;
		std::vector<Eigen::VectorXd> centroids(k, Eigen::VectorXd::Zero(data[0].size()));
		std::vector<int> counts(k, 0);
		for (size_t i = 0; i < data.size(); ++i) {
			int cluster = assignments[i];
			centroids[cluster] += data[i];
			counts[cluster]++;
		}
		for (int i = 0; i < k; ++i) {
			if (counts[i] > 0) {
				centroids[i] /= counts[i];
			}
		}
		for (size_t i = 0; i < data.size(); ++i) {
			int cluster = assignments[i];
			loss += (data[i] - centroids[cluster]).squaredNorm();
		}
		losses[k - 1] = loss / data.size();
	}

	// 寻找肘部
	int bestK = 1;
	double minChange = std::numeric_limits<double>::max();
	for (int k = 1; k < maxK; ++k) {
		double change = losses[k] - losses[k - 1];
		if (change < minChange) {
			minChange = change;
			bestK = k + 1;
		}
	}

	return bestK;
}
double SpectralClustring::LaplacianMatrix::silhouetteScore(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments)
{
	double totalScore = 0.0;

	for (size_t i = 0; i < data.size(); ++i) {
		int cluster = assignments[i];
		double intraClusterDistance = 0.0;
		int numPointsInCluster = 0;

		// 计算样本 i 到同簇其他样本的平均距离（簇内距离）
		for (size_t j = 0; j < data.size(); ++j) {
			if (assignments[j] == cluster && i != j) {
				intraClusterDistance += (data[i] - data[j]).norm();
				numPointsInCluster++;
			}
		}
		if (numPointsInCluster > 0) {
			intraClusterDistance /= numPointsInCluster;
		}

		// 计算样本 i 到其它簇的平均距离（簇间距离）
		double minInterClusterDistance = std::numeric_limits<double>::max();
		for (size_t j = 0; j < data.size(); ++j) {
			if (assignments[j] != cluster) {
				double distance = 0.0;
				int numPointsInOtherCluster = 0;
				for (size_t k = 0; k < data.size(); ++k) {
					if (assignments[k] == assignments[j]) {
						distance += (data[j] - data[k]).norm();
						numPointsInOtherCluster++;
					}
				}
				if (numPointsInOtherCluster > 0) {
					distance /= numPointsInOtherCluster;
					minInterClusterDistance = std::min(minInterClusterDistance, distance);
				}
			}
		}

		// 计算轮廓系数
		double silhouette = 0.0;
		if (intraClusterDistance < minInterClusterDistance) {
			silhouette = 1 - (intraClusterDistance / minInterClusterDistance);
		}
		else if (intraClusterDistance > minInterClusterDistance) {
			silhouette = (minInterClusterDistance / intraClusterDistance) - 1;
		}
		totalScore += silhouette;
	}

	return totalScore / data.size();
}

double SpectralClustring::LaplacianMatrix::calculateClusterLoss(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments)
{
	double totalLoss = 0.0;
	for (size_t i = 0; i < data.size(); ++i) {
		int cluster = assignments[i];
		Eigen::VectorXd centroid(data[0].size());
		int numPointsInCluster = 0;
		for (size_t j = 0; j < data.size(); ++j) {
			if (assignments[j] == cluster) {
				centroid += data[j];
				numPointsInCluster++;
			}
		}
		centroid /= numPointsInCluster;
		for (size_t j = 0; j < data.size(); ++j) {
			if (assignments[j] == cluster) {
				totalLoss += (data[j] - centroid).squaredNorm();
			}
		}
	}
	return totalLoss / data.size();
}

int SpectralClustring::LaplacianMatrix::selectK(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations)
{
	std::vector<double> silhouetteScores(maxK - 1, 0.0);
	std::vector<double> clusterLosses(maxK - 1, 0.0);

	for (int k = 2; k <= maxK; ++k) {
		std::vector<int> assignments = kMeansClustering(data, k, maxIterations);
		silhouetteScores[k - 2] = silhouetteScore(data, assignments);
		clusterLosses[k - 2] = calculateClusterLoss(data, assignments);
	}

	// 计算轮廓系数的均值
	double avgSilhouetteScore = std::accumulate(silhouetteScores.begin(), silhouetteScores.end(), 0.0) / silhouetteScores.size();

	// 计算聚类损失的变化率
	std::vector<double> clusterLossChanges(maxK - 2, 0.0);
	for (int i = 1; i < maxK - 1; ++i) {
		clusterLossChanges[i - 1] = clusterLosses[i] - clusterLosses[i - 1];
	}

	// 根据轮廓系数和聚类损失的变化情况选择最优的 K 值
	int bestK = 2;
	double bestScore = -1.0;
	for (int k = 2; k <= maxK; ++k) {
		double score = silhouetteScores[k - 2] + (clusterLossChanges[k - 3] / clusterLosses[k - 2]);
		if (score > bestScore) {
			bestScore = score;
			bestK = k;
		}
	}

	return bestK;
}

//void SpectralClustring::LaplacianMatrix::plotSilhouetteAndLoss(const std::vector<double>& silhouetteScores, const std::vector<double>& clusterLosses, int maxK)
//{
//	cv::Mat plot = cv::Mat::zeros(400, 800, CV_8UC3);
//	cv::Point2f prevPoint;
//
//	// 绘制轮廓系数曲线
//	for (int k = 2; k <= maxK; ++k) {
//		cv::Point2f point(k * 20, 200 - silhouetteScores[k - 2] * 100);
//		if (k > 2) {
//			cv::line(plot, prevPoint, point, cv::Scalar(255, 0, 0), 2);
//		}
//		prevPoint = point;
//	}
//
//	// 绘制聚类损失曲线
//	prevPoint = cv::Point2f();
//	for (int k = 2; k <= maxK; ++k) {
//		cv::Point2f point(k * 20, 400 - clusterLosses[k - 2] * 100);
//		if (k > 2) {
//			cv::line(plot, prevPoint, point, cv::Scalar(0, 0, 255), 2);
//		}
//		prevPoint = point;
//	}
//
//	cv::imshow("Plot", plot);
//	cv::waitKey(0);
//}

void SpectralClustring::LaplacianMatrix::extractVector()
{
	for (const auto& pair : eigen_pairs) 
	{
		eigenPairs.push_back(pair.second);
	}
}

void SpectralClustring::LaplacianMatrix::kMeansClustering(const std::vector<std::pair<double, Eigen::VectorXd>>& eigen_pairs)
{
	// 提取特征向量
	std::vector<Eigen::VectorXd> data;
	for (const auto& pair : eigen_pairs) {
		data.push_back(pair.second);
	}

	// 选择最佳的 K 值
	int k = selectK(data, 1000, 1000); // 假设最大 K 值为 10，最大迭代次数为 100

	// 使用 KMeans 聚类算法进行聚类
	std::vector<int> assignments = kMeansClustering(data, k, 1000); // 假设最大迭代次数为 100

	// 输出聚类中心 ID
	std::cout << "Cluster centers ID: ";
	for (int i = 0; i < k; ++i) {

		std::cout << i << ": " << assignments[i] << ", ";
	}
	std::cout << std::endl;

	//// 输出每个节点所在的聚类中心 ID
	//std::cout << "Node ID -> Cluster center ID: " << std::endl;
	//for (size_t i = 0; i < data.size(); ++i) {
	//	std::cout << i << " -> " << assignments[i] << std::endl;
	//}


}