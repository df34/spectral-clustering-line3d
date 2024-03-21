#pragma once
#include<iostream>
#include<vector>
//#include<opencv2/opencv.hpp>
#include<Eigen/Sparse>
//#include "rply.h"
#include<fstream>
#include <map>
#include <unordered_map>
using namespace std;
//using namespace cv;
#define RAD2DEG(x) ((x)*180./CV_PI)  //弧度转角度 
#define DEG2RAD(x) ((x)*CV_PI/180.0)  //角度转弧度
#define RESOLUTION 0.0011389


typedef struct Line3D
{
	double X1;
	double Y1;
	double Z1;							//线特征起点三维坐标
	double X2;
	double Y2;
	double Z2;							//线特征终点三维坐标
	int ID;
	//double avgZ;
	Line3D() : X1(0), Y1(0), Z1(0), X2(0), Y2(0), Z2(0), ID(0) {}
	Line3D(double X1, double Y1, double Z1, double X2, double Y2, double Z2,int ID) :X1(X1), Y1(Y1), Z1(Z1), X2(X2), Y2(Y2), Z2(Z2), ID(ID) {};
};

using LinearFeature = struct
{
	double x_midpoint;
	double y_midpoint;
	double z_midpoint;
	double latitude;
	double longitude;
};
typedef struct MinMaxDistance
{
	double minDistance;
	double maxDistance;
};
typedef struct SimilarityBetweenTheLines
{
	double distance;
	double angle;
	double corr;
};


namespace SpectralClustring
{
	class PointSimilarity
	{
	private:
		std::vector<Line3D>lines;
		std::vector<std::vector<double>> similarityMatrix;
		// 创建一个映射，将 serialNumber 映射到索引
		std::map<int, size_t> serialNumberToIndexMap;
		void Grapes();


		double linepearsonCorrelation(Line3D line1, Line3D line2);
		double angleGap(Line3D line1, Line3D line2);
		LinearFeature linearFeature(Line3D line);

		double straightLineDistance(Line3D line1, Line3D line2);
		double pointLineDistance(Line3D line, Line3D point);

		double normalizedDistance(double lineDistance,MinMaxDistance distance);
		double normalizedAngleGap(double angle);
		double normalizedLinepearsonCorrelation(double linCorr);
		MinMaxDistance minMaxDistance();

	public:
		double assignmentSimilarity(Line3D line1,Line3D lin2);
		void updateSimilarityMatrix(Line3D line1, Line3D line2);
		double getSimilarity(int row, int col) const;
	};
	

	class LaplacianMatrix
	{
	private:
		std::vector<Line3D>lines;
		std::vector<std::vector<double>> metricMatrix;
		std::vector<std::vector<double>> LaplacianMatrix;
		std::vector<std::vector<double>> similarityMatrix;
		Eigen::SparseMatrix<double> LaplacianMatrixSparse;
		std::vector<std::pair<double, Eigen::VectorXd>> eigen_pairs;
		std::vector<Eigen::VectorXd> eigenPairs;
		std::vector<int>assignmentID;
		std::unordered_map<int, int> nodeClusterMap;
	public:
		void calculateMetricMatrix(std::vector<std::vector<double>>& metricMatrix);
		void calculateLaplacianMatrix();
		//两种实现方式
		void convertToSparseMatrix(const std::vector<std::vector<double>>& denseMatrix);//稠密矩阵转为稀疏矩阵
		void calSparseMatrix();//稠密矩阵转为稀疏矩阵
		
		void computeEigen();//计算特征值和特征向量
		void fileterEigenPairs();//按照特征值大小进行排序，筛选特征值为0和除0以外最大的特征值

		void extractVector();//只获取特征向量
		std::vector<int> kMeansClustering(const std::vector<Eigen::VectorXd>& data, int k, int maxIterations);
		//肘部法则和轮廓系数
		int selectKByElbowMethod(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations);//返回一个最佳的K值
		double silhouetteScore(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments);//返回轮廓系数，-1到1，距离1越近，效果越好
		//聚类损失
		double calculateClusterLoss(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments);
		//绘制聚类损失曲线-肘部函数
		//void plotSilhouetteAndLoss(const std::vector<double>& silhouetteScores, const std::vector<double>& clusterLosses, int maxK);
		//同时考虑
		int selectK(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations);//返回一个最佳的K值
		void kMeansClustering(const std::vector<std::pair<double, Eigen::VectorXd>>& eigen_pairs);
		void fstreamLine();
	};
	void plyIput();
	void spectralClustringComoleteFlowScheme();
}