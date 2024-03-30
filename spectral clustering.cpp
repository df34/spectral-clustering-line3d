#include"Cluster.h"
#include<numeric>
#include<cmath>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include<Eigen/Eigenvalues>
#include<Eigen/src/Eigenvalues/EigenSolver.h>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include <Spectra/SymEigsSolver.h>
#include<Spectra/MatOp/SparseSymMatProd.h>
#include<Spectra/MatOp/DenseSymMatProd.h>
#include<Spectra/MatOp/SparseSymShiftSolve.h>
#include<Spectra/SymEigsShiftSolver.h>
#include<Spectra/DavidsonSymEigsSolver.h>
#include<random>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<chrono>
#include<limits>

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


//LinearFeature SpectralClustring::PointSimilarity::linearFeature(Line3D line)
//{
//	LinearFeature feature;
//
//	// Calculate midpoint coordinates
//	feature.x_midpoint = (line.X1 + line.X2) / 2.0;
//	feature.y_midpoint = (line.Y1 + line.Y2) / 2.0;
//	feature.z_midpoint = (line.Z1 + line.Z2) / 2.0;
//
//	// Calculate latitude and longitude
//	double radius = std::sqrt(std::pow(feature.x_midpoint, 2) + std::pow(feature.y_midpoint, 2) + std::pow(feature.z_midpoint, 2));
//	feature.latitude = std::asin(feature.z_midpoint / radius) * 180.0 / M_PI;
//	feature.longitude = std::atan2(feature.y_midpoint, feature.x_midpoint) * 180.0 / M_PI;
//
//	return feature;
//}

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
double SpectralClustring::PointSimilarity::guassianSimilarity(Line3D line1, Line3D line2,double sigma)
{
	double distance = straightLineDistance(line1, line2);
	return exp(-distance * distance / (2 * sigma * sigma));
}

double SpectralClustring::PointSimilarity::normalizedDistance(double lineDistance, MinMaxDistance distance)
{
	double minDistance = distance.minDistance;
	double maxDistance = distance.maxDistance;

	// 确保 lineDistance 在最小值和最大值之间
	lineDistance = std::max(minDistance, std::min(maxDistance, lineDistance));

	// 将 lineDistance 归一化到 [0, 1] 范围内
	double normalized = (lineDistance - minDistance) / (maxDistance - minDistance);
	double negetedNormalized = 1.0 - normalized;
	return negetedNormalized;
}


void SpectralClustring::PointSimilarity::minMaxDistance()
{
	double minDist = std::numeric_limits<double>::max();
	double maxDist = std::numeric_limits<double>::lowest();

	for (const auto& line1 : lines)
	{
		for (const auto& line2 : lines) {
			double dist = straightLineDistance(line1,line2);
			minDist = std::min(minDist, dist);
			maxDist = std::max(maxDist, dist);
		}
	}
	minMaxDistances.maxDistance = maxDist;
	minMaxDistances.minDistance = minDist;
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
	double negetedNormalized = 1.0 - normalized;
	return negetedNormalized;
}
double SpectralClustring::PointSimilarity::normalizedLinepearsonCorrelation(double linCorr)
{
	double normalized = (linCorr + 1.0) / 2.0; // 将 [-1, 1] 映射到 [0, 1]
	return normalized;
}


double SpectralClustring::PointSimilarity::assignmentSimilarity(Line3D line1, Line3D line2)
{
	//double p1 = 0.4, p2 = 0.3, p3 = 0.3;
	double similarity = 0;
	double angle = angleGap(line1, line2);
	if (angle < M_PI / 24)
	{
		//double lineDistance = straightLineDistance(line1, line2);
		double corr = linepearsonCorrelation(line1, line2);
		similarity = normalizedDistance(guassianSimilarity(line1, line2, 0.1), minMaxDistances);
		//similarity = p1 * normalizedDistance(guassianSimilarity(line1,line2,0.1), minMaxDistances) + p2 * normalizedAngleGap(angle) + p3 * normalizedLinepearsonCorrelation(corr);
	}
	else
	{
		similarity = 0;
	}
	return similarity;
}

void SpectralClustring::PointSimilarity::updateSimilarityMatrix(Line3D line1, Line3D line2)
{
	int id1 = line1.ID;
	int id2 = line2.ID;

	// 检查矩阵中是否已经存在相似度值，如果存在则不进行计算
	if (id1 < similarityMatrix.size() && id2 < similarityMatrix[id1].size() && similarityMatrix[id1][id2] != 0.0)
	{
		return;
	}

	double similarity = assignmentSimilarity(line1, line2);

	// 确保 similarityMatrix 具有足够的大小
	int newSize = std::max(id1, id2) + 1;
	if (similarityMatrix.size() < newSize)
	{
		// 扩展行
		for (int i = 0; i < similarityMatrix.size(); ++i)
		{
			similarityMatrix[i].resize(newSize, 0.0);
		}
		// 扩展列
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

void SpectralClustring::LaplacianMatrix::calculateMetricMatrix()
{
	metricMatrix.clear();
	metricMatrix.resize(similarityMatrix.size(), std::vector<double>(similarityMatrix.size(), 0.0));

	// 计算度矩阵
	for (size_t i = 0; i < similarityMatrix.size(); ++i)
	{
		double sum = 0.0;
		for (size_t j = 0; j < similarityMatrix[i].size(); ++j)
		{
			sum += similarityMatrix[i][j];
		}
		metricMatrix[i][i] = sum;
	}
}
void SpectralClustring::LaplacianMatrix::calculateNormalizedLaplacianMatrix()
{
	std::vector<std::vector<double>>normalizedLaplacianMatrixs;
	normalizedLaplacianMatrixs.clear();
	normalizedLaplacianMatrixs.resize(similarityMatrix.size(), std::vector<double>(similarityMatrix.size(), 0.0));
	//normalizedLaplacianMatrixs.assign(similarityMatrix.size(), std::vector<double>(similarityMatrix.size(), 0.0));
	// 计算度矩阵和未标准化拉普拉斯矩阵
	std::vector<double> degrees(similarityMatrix.size(), 0.0);
	std::vector<std::vector<double>> unnormalizedLaplacian(similarityMatrix.size(), std::vector<double>(similarityMatrix.size(), 0.0));

	for (size_t i = 0; i < similarityMatrix.size(); ++i)
	{
		for (size_t j = 0; j < similarityMatrix[i].size(); ++j)
		{
			degrees[i] += similarityMatrix[i][j];
			unnormalizedLaplacian[i][j] = -similarityMatrix[i][j];
		}
		unnormalizedLaplacian[i][i] = degrees[i];
	}

	// 计算 D^{-1/2}
	std::vector<double> D_sqrt_inv(degrees.size());
	for (size_t i = 0; i < degrees.size(); ++i)
	{
		// 添加容错机制，确保分母不为零
		if (degrees[i] > 1e-6) {
			D_sqrt_inv[i] = 1.0 / std::sqrt(degrees[i]);
		}
		else {
			D_sqrt_inv[i] = 0.0;  // 或者使用其他合适的值来避免除以零
		}
	}

	// 计算标准化后的拉普拉斯矩阵 D^{-1/2} L D^{-1/2}
	for (size_t i = 0; i < similarityMatrix.size(); ++i)
	{
		for (size_t j = 0; j < similarityMatrix[i].size(); ++j)
		{
			normalizedLaplacianMatrixs[i][j] = D_sqrt_inv[i] * unnormalizedLaplacian[i][j] * D_sqrt_inv[j];
		}
	}
	LaplacianMatrixs.clear();
	LaplacianMatrixs.resize(normalizedLaplacianMatrixs.size(), std::vector<double>(normalizedLaplacianMatrixs.size(), 0.0));
	for (size_t i = 0; i < normalizedLaplacianMatrixs.size(); ++i) {
		for (size_t j = 0; j < normalizedLaplacianMatrixs[i].size(); ++j) {
			LaplacianMatrixs[i][j] = normalizedLaplacianMatrixs[i][j];
		}
	}
}

void SpectralClustring::LaplacianMatrix::calculateLaplacianMatrix()
{
	LaplacianMatrixs.clear();
	LaplacianMatrixs.resize(metricMatrix.size(), std::vector<double>(metricMatrix.size(), 0.0));

	// 计算拉普拉斯矩阵
	for (size_t i = 0; i < metricMatrix.size(); ++i)
	{
		double degree = 0.0; // 节点的度数
		for (size_t j = 0; j < metricMatrix[i].size(); ++j)
		{
			degree += metricMatrix[i][j];
		}

		for (size_t j = 0; j < metricMatrix[i].size(); ++j)
		{
			if (i == j)
			{
				// 对角线元素为节点的度数
				LaplacianMatrixs[i][j] = degree;
			}
			else
			{
				// 非对角线元素为相似度或连接权重的负值
				LaplacianMatrixs[i][j] = -metricMatrix[i][j];
			}
		}
	}
	std::cout << "end";

}

//void SpectralClustring::LaplacianMatrix::convertToSparseMatrix()
//{
//	if (LaplacianMatrixs.empty()) {
//		// 处理异常情况，如 denseMatrix 为空
//		return;
//	}
//
//	int rows = LaplacianMatrixs.size();
//	int cols = LaplacianMatrixs[0].size();
//
//	// 遍历稠密矩阵，将非零元素添加到稀疏矩阵中
//	std::vector<Triplet<double>> triplets;
//	for (int i = 0; i < rows; ++i)
//	{
//		if (LaplacianMatrixs[i].size() != cols) {
//			// 处理异常情况，如 denseMatrix 中的行长度不一致
//			return;
//		}
//
//		for (int j = 0; j < cols; ++j)
//		{
//			if (LaplacianMatrixs[i][j] != 0.0)
//			{
//				triplets.push_back(Triplet<double>(i, j, LaplacianMatrixs[i][j]));
//			}
//		}
//	}
//
//	// 创建一个稀疏矩阵
//	SparseMatrix<double> sparseMatrix(rows, cols);
//	sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
//
//	LaplacianMatrixSparse = sparseMatrix;
//}

void SpectralClustring::LaplacianMatrix::convertToSparseMatrixPlus()
{

}

//void SpectralClustring::LaplacianMatrix::calSparseMatrix()
//{
//	// 创建稀疏矩阵的三元组表示
//	std::vector<Eigen::Triplet<double>> tripletList;
//
//	// 遍历密集矩阵的元素，将非零元素添加到三元组列表中
//	for (int i = 0; i < LaplacianMatrixs.size(); ++i) {
//		for (int j = 0; j < LaplacianMatrixs[i].size(); ++j) {
//			if (LaplacianMatrixs[i][j] != 0.0) {
//				tripletList.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrixs[i][j]));
//			}
//		}
//	}
//
//	// 将三元组列表设置为稀疏矩阵的值
//	LaplacianMatrixSparse.setFromTriplets(tripletList.begin(), tripletList.end());
//
//	// 完成矩阵插入后进行最终化
//	LaplacianMatrixSparse.finalize();
//}

void SpectralClustring::LaplacianMatrix::computeEigen()
{
	/*EigenSolver<SpMat>es(LaplacianMatrixSparse);
	VectorXcd eigenvalues = es.eigenvalues();
	MatrixXcd eigenvectors = es.eigenvectors();*/
	SparseSymMatProd<double>op(LaplacianMatrixSparse);
	std::cout << LaplacianMatrixSparse.rows();
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

void SpectralClustring::LaplacianMatrix::computeEigenPlus()
{
	if (LaplacianMatrixs.empty()) {
		// 处理异常情况，如 LaplacianMatrixs 为空
		return;
	}

	int rows = LaplacianMatrixs.size();
	int cols = LaplacianMatrixs[0].size();

	// 将 LaplacianMatrixs 转换为稀疏矩阵 LaplacianMatrixSparse
	if (LaplacianMatrixSparse.rows() != rows || LaplacianMatrixSparse.cols() != cols) {
		// 重新分配稀疏矩阵大小
		LaplacianMatrixSparse.resize(rows, cols);
	}

	// 遍历稠密矩阵，将非零元素添加到稀疏矩阵中
	std::vector<Triplet<double>> triplets;
	for (int i = 0; i < rows; ++i) {
		if (LaplacianMatrixs[i].size() != cols) {
			// 处理异常情况，如 denseMatrix 中的行长度不一致
			return;
		}

		for (int j = 0; j < cols; ++j) {
			if (LaplacianMatrixs[i][j] != 0.0) {
				triplets.push_back(Triplet<double>(i, j, LaplacianMatrixs[i][j]));
			}
		}
	}

	// 创建一个稀疏矩阵
	LaplacianMatrixSparse.setFromTriplets(triplets.begin(), triplets.end());

	// 使用 SymEigsSolver 求解特征值和特征向量
	SparseSymMatProd<double> op(LaplacianMatrixSparse);
	Spectra::SymEigsSolver< Spectra::SparseSymMatProd<double> > eigs(op, 2, 2 * LaplacianMatrixSparse.rows());
	eigs.init();
	int nconv = eigs.compute();

	// 获取特征值
	Eigen::VectorXd eigenvalues;
	if (eigs.info() == Spectra::CompInfo::Successful) {
		eigenvalues = eigs.eigenvalues();
	}

	// 获取特征向量
	Eigen::MatrixXd eigenvectors;
	if (eigs.info() == Spectra::CompInfo::Successful) {
		eigenvectors = eigs.eigenvectors();
	}

	// 将特征值和特征向量存储到一个键对的数组中
	for (int i = 0; i < eigenvalues.size(); ++i) {
		eigen_pairs.push_back(std::make_pair(eigenvalues[i], eigenvectors.col(i)));
	}

	// 打印特征值和特征向量
	for (const auto& pair : eigen_pairs) {
		std::cout << "Eigenvalue: " << pair.first << std::endl;
		std::cout << "Eigenvector: " << pair.second.transpose() << std::endl;
	}
}
void SpectralClustring::LaplacianMatrix::computeEigenPlusPlus()
{
	std::vector<Eigen::Triplet<double>> triplets;
	for (int i = 0; i < LaplacianMatrixs.size(); ++i) {
		for (int j = 0; j < LaplacianMatrixs[i].size(); ++j) {
			if (LaplacianMatrixs[i][j] != 0.0) {
				triplets.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrixs[i][j]));
			}
		}
	}
	Eigen::SparseMatrix<double> LaplacianMatrixSparses(LaplacianMatrixs.size(), LaplacianMatrixs.size());
	LaplacianMatrixSparses.setFromTriplets(triplets.begin(), triplets.end());
	Spectra::SparseSymMatProd<double>op(LaplacianMatrixSparses);
	int k = LaplacianMatrixSparses.rows() - 2;
	int m = LaplacianMatrixSparses.rows() - 1;
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;
	Spectra::SymEigsSolver< Spectra::SparseSymMatProd<double> > eigs(op, k, m);
	eigs.init();
	int nconv = eigs.compute();
	if (eigs.info() == Spectra::CompInfo::Successful) {
		eigenvalues = eigs.eigenvalues();
		eigenvectors = eigs.eigenvectors();
	}
	// 将特征值和特征向量存储到一个键对的数组中
	for (int i = 0; i < eigenvalues.size(); ++i) {
		eigen_pairs.push_back(std::make_pair(eigenvalues[i], eigenvectors.col(i)));
	}
	// 打印特征值和特征向量
	
	for (const auto& pair : eigen_pairs) {
		std::cout << "Eigenvalue: " << pair.first << std::endl;
		std::cout << "Eigenvector: " << pair.second.transpose() << std::endl;
	}
	
}
void SpectralClustring::LaplacianMatrix::computeEigenRandomized()
{
	std::vector<Eigen::Triplet<double>> triplets;
	for (int i = 0; i < LaplacianMatrixs.size(); ++i) {
		for (int j = 0; j < LaplacianMatrixs[i].size(); ++j) {
			if (LaplacianMatrixs[i][j] != 0.0) {
				triplets.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrixs[i][j]));
			}
		}
	}

	Eigen::SparseMatrix<double> LaplacianMatrixSparse(LaplacianMatrixs.size(), LaplacianMatrixs.size());
	LaplacianMatrixSparse.setFromTriplets(triplets.begin(), triplets.end());

	// 随机化算法的参数
	int subspace_dim = 10; // 子空间的维度，可以根据需要调整

	// 创建随机数生成器
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	// 生成随机矩阵
	Eigen::MatrixXd random_matrix(LaplacianMatrixSparse.rows(), subspace_dim);
	for (int i = 0; i < random_matrix.rows(); ++i) {
		for (int j = 0; j < random_matrix.cols(); ++j) {
			random_matrix(i, j) = distribution(generator);
		}
	}

	// 计算投影矩阵
	Eigen::MatrixXd projection = LaplacianMatrixSparse * random_matrix;

	// 计算投影矩阵的 SVD
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(projection, Eigen::ComputeThinU | Eigen::ComputeThinV);

	// 提取近似特征向量
	Eigen::MatrixXd eigenvectors = svd.matrixU();

	// 计算近似特征值
	Eigen::VectorXd eigenvalues = svd.singularValues();

	// 打印近似特征值和特征向量
	for (int i = 0; i < eigenvalues.size(); ++i) {
		std::cout << "近似特征值: " << eigenvalues[i] << std::endl;
		std::cout << "近似特征向量: " << eigenvectors.col(i).transpose() << std::endl;
	}
}
void SpectralClustring::LaplacianMatrix::computeEigenShift()
{
	/*std::vector<Eigen::Triplet<double>> triplets;
	for (int i = 0; i < LaplacianMatrixs.size(); ++i) {
		for (int j = 0; j < LaplacianMatrixs[i].size(); ++j) {
			if (LaplacianMatrixs[i][j] != 0.0) {
				triplets.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrixs[i][j]));
			}
		}
	}*/

	std::vector<Eigen::Triplet<double>> triplets;
	for (int i = 0; i < LaplacianMatrixs.size(); ++i) {
		for (int j = 0; j < LaplacianMatrixs[i].size(); ++j) {
			if (LaplacianMatrixs[i][j] != 0.0) {
				triplets.push_back(Eigen::Triplet<double>(i, j, LaplacianMatrixs[i][j]));
			}
		}
	}
	/*int rows = LaplacianMatrixs.size();
	int cols = LaplacianMatrixs[0].size();
	MatrixXd matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			matrix(i, j) = LaplacianMatrixs[i][j];
	}
	DenseSymShiftSolve<double>op(matrix);*/

	//Eigen::SparseMatrix<double> LaplacianMatrixSparse(LaplacianMatrixs.size(), LaplacianMatrixs.size());
	Eigen::SparseMatrix<double> LaplacianMatrixSparse(LaplacianMatrixs.size(), LaplacianMatrixs.size());
	LaplacianMatrixSparse.setFromTriplets(triplets.begin(), triplets.end());

	// 使用 Spectra 的移位和反转策略
	Spectra::SparseSymShiftSolve<double> op(LaplacianMatrixSparse);
	double sigma = 1e-10; // 移位量，接近零的小正数
	int nev = 500;
	int ncv = std::max(2 * nev, 20);
	Spectra::SymEigsShiftSolver< Spectra::SparseSymShiftSolve<double> > eigs(op, nev, ncv, sigma);

	// 初始化和计算
	eigs.init();
	int nconv = eigs.compute(Spectra::SortRule::LargestAlge);//寻找最小的特征值

	// 检查结果是否成功
	if (eigs.info() == Spectra::CompInfo::Successful) {
		Eigen::VectorXd eigenvalues = eigs.eigenvalues();
		Eigen::MatrixXd eigenvectors = eigs.eigenvectors();

		// 存储特征值和特征向量
		for (int i = 0; i < eigenvalues.size(); ++i) {
			eigen_pairs.push_back(std::make_pair(eigenvalues[i], eigenvectors.col(i)));
		}

		// 打印特征值和特征向量
		/*for (const auto& pair : eigen_pairs) {
			std::cout << "Eigenvalue: " << pair.first << std::endl;
			std::cout << "Eigenvector: " << pair.second.transpose() << std::endl;
		}*/
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
void SpectralClustring::LaplacianMatrix::normalizeEigenPairs()
{
	for (size_t i = 0; i < eigenPairs.size(); ++i)
	{
		double norm = eigenPairs[i].norm();
		eigenPairs[i] /= norm;
	}
}
void SpectralClustring::LaplacianMatrix::storeEigenPairs()
{
	F.clear();
	F.reserve(eigenPairs.size());
	for (size_t i = 0; i < eigenPairs.size(); ++i)
	{
		F.push_back(eigenPairs[i]);
	}
}
std::vector<int> SpectralClustring::LaplacianMatrix::kMeansClustering(const std::vector<Eigen::VectorXd>& data, const std::vector<Eigen::VectorXd>& centroids, int maxIterations)
{
	/*if (data.empty() || centroids.empty() || centroids.size() >= data.size() || maxIterations <= 0) {
		throw std::invalid_argument("Invalid arguments for kMeansClustering.");
	}
	std::cout << "done" << std::endl;*/
	bool changed = true;
	std::vector<int> assignments(data.size(), 0); // 使用data的大小来初始化assignments
	
	int iter = 0;
	std::vector<Eigen::VectorXd> updatedCentroids(centroids.begin(), centroids.end()); // 创建一个可修改的副本

	while (iter < maxIterations && changed) {
		changed = false;
		// 分配样本到最近的聚类中心
		for (size_t i = 0; i < data.size(); ++i) {
			double minDistance = std::numeric_limits<double>::max();
			int clusterIndex = -1;
			for (size_t j = 0; j < centroids.size(); ++j) {
				double distance = (data[i] - centroids[j]).squaredNorm();
				if (distance < minDistance) {
					minDistance = distance;
					clusterIndex = static_cast<int>(j);
				}
			}
			if (assignments[i] != clusterIndex) {
				assignments[i] = clusterIndex;
				changed = true;
			}
		}

		// 更新聚类中心
		std::vector<Eigen::VectorXd> newCentroids(centroids.size(), Eigen::VectorXd::Zero(data[0].size()));
		std::vector<int> counts(centroids.size(), 0);
		for (size_t i = 0; i < assignments.size(); ++i) {
			newCentroids[assignments[i]] += data[i];
			counts[assignments[i]]++;
		}
		for (size_t j = 0; j < centroids.size(); ++j) {
			if (counts[j] > 0) {
				updatedCentroids[j] = (newCentroids[j].array() / static_cast<double>(counts[j])).matrix();
			}
		}
		iter++;
	}

	return assignments;
}
std::vector<Eigen::VectorXd> SpectralClustring::LaplacianMatrix::kMeansClusteringPlus(const std::vector<Eigen::VectorXd>& data, int k)
{
	std::vector<Eigen::VectorXd> centroids;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, data[0].size() - 1); // Assuming all rows have the same size

	// Randomly select the first centroid
	Eigen::VectorXd firstCentroid(data[0].size());
	//Eigen::VectorXd database(data[0].size());
	/*if (!data.empty())
	{
		for (int j=0;j<data.size();j++)
		{
			for (int i = 0; i < data[0].size(); i++)
			{
				database[j] += data[j][i];
			}
		}
	}*/
	for (int i = 0; i < data.size(); i++)
	{
		int index = dis(gen);
		if (index >= data.size())
		{
			std::cout << "Error: Index out of range: " << index << std::endl;
			break;
		}
		firstCentroid(i) = data[index](i);
	}
	
	centroids.push_back(firstCentroid);

	// Select the remaining centroids
	for (int i = 1; i < k; ++i) {
		// Calculate the distance of each point to the nearest centroid
		std::vector<double> distances(data.size(), std::numeric_limits<double>::max());
		for (size_t j = 0; j < data.size(); ++j) {
			for (const auto& centroid : centroids) {
				double dist = (data[j] - centroid).squaredNorm();
				distances[j] = std::min(distances[j], dist);
			}
		}

		// Select the next centroid based on the distances
		double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
		std::uniform_real_distribution<> distr(0, sum);
		double r = distr(gen);

		sum = 0.0;
		for (size_t j = 0; j < distances.size(); ++j) {
			sum += distances[j];
			if (sum >= r) {
				centroids.push_back(data[j]);
				break;
			}
		}
	}
	return centroids;
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
				double distance = (data[i] - data[j]).norm();
				intraClusterDistance += distance;
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
	const double minClusterLossThreshold = 0.001; // 设定一个较小的阈值
	for (int k = 4; k <= maxK; ++k) {
		std::vector<Eigen::VectorXd> centroids = kMeansClusteringPlus(data, k);
		std::vector<int> assignments = kMeansClustering(data, centroids, maxIterations);
		silhouetteScores[k - 2] = silhouetteScore(data, assignments);
		clusterLosses[k - 2] = calculateClusterLoss(data, assignments);
		// 如果聚类损失小于阈值，跳出循环
		if (clusterLosses[k - 2] < minClusterLossThreshold) {
			break;
		}
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
	for (int k = 2; k < maxK; ++k) {
		if (clusterLosses[k - 1] != 0.0) {
			double score = silhouetteScores[k - 2] + (clusterLossChanges[k - 2] / clusterLosses[k - 1]);
			if (score > bestScore) {
				bestScore = score;
				bestK = k;
			}
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

std::vector<int> SpectralClustring::LaplacianMatrix::kMeansClustering(const std::vector<Eigen::VectorXd>& data, int k, int maxIterations)
{
	if (data.empty() || k <= 0 || k > data.size() || maxIterations <= 0) {
		throw std::invalid_argument("Invalid arguments for kMeansClustering.");
	}
	// 初始化聚类中心
	std::vector<Eigen::VectorXd> centroids= kMeansClusteringPlus(data, k);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, data.size() - 1);
	for (int i = 0; i < k; ++i) {
		centroids[i] = data[dist(gen)];
	}
	// 开始迭代
	std::vector<int> assignments(data.size(), -1);
	bool changed = true;
	int iter = 0;
	while (iter < maxIterations && changed) {
		changed = false;
		// 分配样本到最近的聚类中心
		for (size_t i = 0; i < data.size(); ++i) {
			double minDistance = std::numeric_limits<double>::max();
			int clusterIndex = -1;
			for (int j = 0; j < k; ++j) {
				double distance = (data[i] - centroids[j]).squaredNorm();
				if (distance < minDistance) {
					minDistance = distance;
					clusterIndex = j;
				}
			}
			if (assignments[i] != clusterIndex)
			{
				assignments[i] = clusterIndex;
				changed = true;
			}
		}
	}
}

void SpectralClustring::LaplacianMatrix::extractVector()
{
	for (const auto& pair : eigen_pairs) 
	{
		eigenPairs.push_back(pair.second);
	}
}

void SpectralClustring::LaplacianMatrix::kMeansClustering()
{
	// 提取特征向量
	std::vector<Eigen::VectorXd> data;
	data.resize(F.size());
	/*for (const auto& pair : eigen_pairs) {
		data.push_back(pair.second);
	}*/
	std::copy(F.begin(), F.end(), data.begin());
	// 选择最佳的 K 值
	//int k = selectKByElbowMethod(data, 100, 100);
	int k = selectK(data, 100, 100); // 假设最大 K 值为 10，最大迭代次数为 100
	std::vector<Eigen::VectorXd>data1;
	for (const auto& pair : eigen_pairs) {
		data1.push_back(pair.second);
	}
	// 使用 KMeans 聚类算法进行聚类
	std::vector<int> assignments = kMeansClustering	(data, k, 100); // 假设最大迭代次数为 100

	// 输出聚类中心 ID
	std::cout << "Cluster centers ID: ";
	for (int i = 0; i < k; ++i) {

		std::cout << i << ": " << assignments[i] << ", ";
	}
	std::cout << std::endl;

	//每个节点的ID和其所在的聚类中心ID存储为一个键值对
	for (size_t i = 0; i < data.size(); ++i) 
	{
		nodeClusterMap[i] = assignments[i];
	}
	// 输出每个节点所在的聚类中心 ID
	std::cout << "Node ID -> Cluster center ID: " << std::endl;
	for (const auto& pair : nodeClusterMap) {
		std::cout << pair.first << " -> " << pair.second << std::endl;
	}
	//// 输出每个节点所在的聚类中心 ID
	//std::cout << "Node ID -> Cluster center ID: " << std::endl;
	//for (size_t i = 0; i < data.size(); ++i) {
	//	std::cout << i << " -> " << assignments[i] << std::endl;
	//}
}

void SpectralClustring::LaplacianMatrix::fstreamLine()
{
	std::ofstream outputFile("cluster_centers.txt");
	if (!outputFile.is_open()) {
		std::cerr << "Failed to open output file." << std::endl;
		return;
	}
	for (int centerID : assignmentID)
	{
		for (const Line3D& line : lines)
		{
			if (line.ID == centerID)
			{
				outputFile << "Cluster Center ID: " << centerID << std::endl;
				outputFile << "Line ID: " << line.ID << std::endl;
				outputFile << "Line Start: (" << line.X1 << ", " << line.Y1 << ", " << line.Z1 << ")" << std::endl;
				outputFile << "Line End: (" << line.X2 << ", " << line.Y2 << ", " << line.Z2 << ")" << std::endl;
				outputFile << std::endl;
				break;
			}
		}
	}
	outputFile.close();




}

std::vector<Line3D> SpectralClustring::readLinesFromPLY(const std::string& filename)
{
	std::vector<Line3D> lines;
	std::ifstream inputFile(filename);
	if (!inputFile.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return lines;
	}

	std::string line;
	int id = 0;
	Point3D prevPoint;
	bool isFirstPoint = true;

	while (std::getline(inputFile, line)) {
		std::istringstream iss(line);
		if (isFirstPoint) {
			if (line.find("element vertex") != std::string::npos) {
				int numVertices;
				iss >> numVertices;
				id = 0;
			}
			else if (line == "end_header") {
				isFirstPoint = false;
			}
		}
		else {
			Point3D point;
			iss >> point.X >> point.Y >> point.Z;
			if (id % 2 != 0) { // Connect odd IDs to form lines
				Line3D line3d;
				line3d.X1 = prevPoint.X;
				line3d.Y1 = prevPoint.Y;
				line3d.Z1 = prevPoint.Z;
				line3d.X2 = point.X;
				line3d.Y2 = point.Y;
				line3d.Z2 = point.Z;
				line3d.ID = id / 2; // ID for the line
				lines.push_back(line3d);
			}
			prevPoint = point;
			++id;
		}
	}

	inputFile.close();
	return lines;
}
bool SpectralClustring::readPointFromPLY(const std::string& filename, std::vector<Vertex>& vertices)
{
	std::ifstream file(filename);
	if (!file.is_open()) {

		std::cout << "Failed to open file: " << filename << std::endl;

		return false;
	}

	std::string line;
	std::getline(file, line); // Read the first line

	// Read header
	int numVertices = 0;
	while (line.find("end_header") == std::string::npos) {
		std::getline(file, line);

		if (line.find("element vertex") != std::string::npos) {
			std::istringstream iss(line);
			std::string element, vertex;
			int count;

			iss >> element >> vertex >> count;
			numVertices = count;
		}
	}

	// Read vertices
	vertices.resize(numVertices);
	for (int i = 0; i < numVertices; ++i) {
		file >> vertices[i].x >> vertices[i].y >> vertices[i].z
			>> vertices[i].r >> vertices[i].g >> vertices[i].b;
	}

	file.close();
	return true;
}

void SpectralClustring::LaplacianMatrix::writeLinesToPLY(const std::string& filename, const std::vector<Line3D>& lines, const std::vector<int>& centerIDs)
{
	std::ofstream outputFile(filename);

	if (!outputFile.is_open())
	{
		std::cerr << "Failed to open file for writing: " << filename << std::endl;
		return;
	}

	// 写入PLY文件头部
	outputFile << "ply" << std::endl;
	outputFile << "format ascii 1.0" << std::endl;
	outputFile << "element vertex " << centerIDs.size() * 2 << std::endl;
	outputFile << "property float x" << std::endl;
	outputFile << "property float y" << std::endl;
	outputFile << "property float z" << std::endl;
	outputFile << "element edge " << centerIDs.size() << std::endl;
	outputFile << "property int vertex1" << std::endl;
	outputFile << "property int vertex2" << std::endl;
	outputFile << "end_header" << std::endl;

	// 写入线的顶点信息
	for (size_t i = 0; i < centerIDs.size(); ++i)
	{
		for (const Line3D& line : lines)
		{
			if (line.ID == centerIDs[i])
			{
				outputFile << line.X1 << " " << line.Y1 << " " << line.Z1 << std::endl;
				outputFile << line.X2 << " " << line.Y2 << " " << line.Z2 << std::endl;
				break;
			}
		}
	}

	// 写入线的连接信息
	for (size_t i = 0; i < centerIDs.size(); ++i)
	{
		outputFile << i * 2 << " " << i * 2 + 1 << std::endl;
	}

	outputFile.close();
}

//std::vector<Line3D> SpectralClustring::LaplacianMatrix::outPutCentreLine()
//{
//	for (int targetID : assignmentID)
//	{
//		try {
//			Line3D foundLine = findLineByID(targetID);
//
//		}
//	}
//}
//Line3D SpectralClustring::LaplacianMatrix::findLineByID(int targetID) const
//{
//	for (const Line3D& line : lines) {
//		if (line.ID == targetID) {
//			return line;
//		}
//	}
//	// 如果没有找到匹配的ID，则可以返回一个默认的Line3D对象或抛出异常，具体取决于你的需求
//	throw std::runtime_error("Line with ID " + std::to_string(targetID) + " not found.");
//}


std::vector<std::vector<double>> SpectralClustring::spectralClustringComoleteFlowScheme(std::vector<Line3D>inputLines)
{
	auto start = std::chrono::high_resolution_clock::now();
	SpectralClustring::PointSimilarity lineSimilarityMatrix;
	lineSimilarityMatrix.getLine(inputLines);
	lineSimilarityMatrix.minMaxDistance();
	for (const auto& line1 : lineSimilarityMatrix.outLine())
	{
		for (const auto& line2 : lineSimilarityMatrix.outLine())
		{
			lineSimilarityMatrix.updateSimilarityMatrix(line1,line2);
		}
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Calculate Similarity Matrix execution time: " << duration.count() << " seconds" << std::endl;
	return lineSimilarityMatrix.outPutSimilarity();
}

void SpectralClustring::PointSimilarity::getLine(std::vector<Line3D>inputLines)
{
	lines = inputLines;
}
std::vector<Line3D> SpectralClustring::PointSimilarity::outLine()
{
	return lines;
}
std::vector<std::vector<double>> SpectralClustring::PointSimilarity::outPutSimilarity()
{
	return similarityMatrix;
}

void SpectralClustring::LaplacianMatrix::calcuCluster()
{
	/*auto start1 = std::chrono::high_resolution_clock::now();
	calculateMetricMatrix();
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration1 = end1 - start1;
	std::cout << "Calculate metric matrix execution time: " << duration1.count() << " seconds" << std::endl;*/
	auto start2 = std::chrono::high_resolution_clock::now();
	//calculateLaplacianMatrix();
	calculateNormalizedLaplacianMatrix();
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration2 = end2 - start2;
	std::cout << "Calculate LaplacianMatrix matrix execution time: " << duration2.count() << " seconds" << std::endl;
	//convertToSparseMatrix();
	auto start3 = std::chrono::high_resolution_clock::now();
	computeEigenShift();
	auto end3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration3 = end3 - start3;
	std::cout << "Calculate eigenvalues eigenvectors execution time: " << duration3.count() << " seconds" << std::endl;
	fileterEigenPairs();
	extractVector();
	normalizeEigenPairs();
	storeEigenPairs();
	auto start4 = std::chrono::high_resolution_clock::now();
	kMeansClustering();
	auto end4 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration4 = end4 - start4;
	std::cout << "Calculate Kmeans execution time: " << duration4.count() << " seconds" << std::endl;
	fstreamLine();
	string filename = "cluster";
	writeLinesToPLY(filename,lines,assignmentID);
}

std::vector<Line3D> SpectralClustring::LaplacianMatrix::outLines()
{
	return lines;
}
std::vector<int> SpectralClustring::LaplacianMatrix::outID()
{
	return assignmentID;
}
void SpectralClustring::LaplacianMatrix::getSimilarrityMatrix(std::vector<std::vector<double>>a)
{
	similarityMatrix = a;
}