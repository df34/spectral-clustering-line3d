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

struct Point3D
{
	double X;
	double Y;
	double Z;
};
struct Vertex
{
	double x, y, z;
	int r, g, b;
};

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
		void Grapes();//线段ID存储


		double linepearsonCorrelation(Line3D line1, Line3D line2);//直线间相似性计算：-1至1，越接近1越相似
		double angleGap(Line3D line1, Line3D line2);//直线间角度差计算，返回弧度值
		//LinearFeature linearFeature(Line3D line);//计算单条直线在坐标系下的中点坐标，与x轴y轴的夹角

		double straightLineDistance(Line3D line1, Line3D line2);//计算直线间距离
		double pointLineDistance(Line3D line, Line3D point);//如果平行，则计算点与直线的距离

		double normalizedDistance(double lineDistance,MinMaxDistance distance);//归一化距离,取反
		double normalizedAngleGap(double angle);//归一化角度差，取反
		double normalizedLinepearsonCorrelation(double linCorr);//归一化直线相似性
		MinMaxDistance minMaxDistances;
		double guassianSimilarity(Line3D line1, Line3D line2,double sigma);

	public:
		void getLine(std::vector<Line3D>);
		std::vector<Line3D>outLine();
		double assignmentSimilarity(Line3D line1,Line3D lin2);//综合计算直线间的相似性系数
		void updateSimilarityMatrix(Line3D line1, Line3D line2);//更新相似矩阵
		double getSimilarity(int row, int col) const;//获取对应行列相似矩阵的值
		std::vector<std::vector<double>>outPutSimilarity();
		void minMaxDistance();//计算所有空间直线的最大最小距离
	};
	

	class LaplacianMatrix
	{
	private:
		std::vector<Line3D>lines;
		std::vector<std::vector<double>> metricMatrix;
		std::vector<std::vector<double>> LaplacianMatrixs;
		std::vector<std::vector<double>> similarityMatrix;
		
		Eigen::SparseMatrix<double> LaplacianMatrixSparse;
		std::vector<std::pair<double, Eigen::VectorXd>> eigen_pairs;
		std::vector<Eigen::VectorXd> eigenPairs;
		std::vector<Eigen::VectorXd> F;
		std::vector<int>assignmentID;//聚类中心ID
		std::unordered_map<int, int> nodeClusterMap;//每个节点所在的聚类中心
		//std::vector<Line3D> outPutCentreLine();
	public:
		void calculateMetricMatrix();//计算度矩阵
		void calculateLaplacianMatrix();//计算拉普拉斯矩阵
		//两种实现方式
		//void convertToSparseMatrix();//稠密矩阵转为稀疏矩阵
		//void calSparseMatrix();//稠密矩阵转为稀疏矩阵
		
		void computeEigen();//计算特征值和特征向量
		void fileterEigenPairs();//按照特征值大小进行排序，筛选特征值为0和除0以外最大的特征值

		void extractVector();//只获取特征向量
		std::vector<int> kMeansClustering(const std::vector<Eigen::VectorXd>& data, const std::vector<Eigen::VectorXd>& centroids, int maxIterations);
		std::vector<int> kMeansClustering(const std::vector<Eigen::VectorXd>& data, int k, int maxIterations);
		std::vector<Eigen::VectorXd> kMeansClusteringPlus(const std::vector<Eigen::VectorXd>& data, int k);
		//肘部法则和轮廓系数
		int selectKByElbowMethod(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations);//返回一个最佳的K值
		double silhouetteScore(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments);//返回轮廓系数，-1到1，距离1越近，效果越好
		//聚类损失
		double calculateClusterLoss(const std::vector<Eigen::VectorXd>& data, const std::vector<int>& assignments);
		//绘制聚类损失曲线-肘部函数
		//void plotSilhouetteAndLoss(const std::vector<double>& silhouetteScores, const std::vector<double>& clusterLosses, int maxK);
		//同时考虑
		int selectK(const std::vector<Eigen::VectorXd>& data, int maxK, int maxIterations);//返回一个最佳的K值
		void kMeansClustering();
		void fstreamLine();//输出聚类中心的点的坐标
		void calcuCluster();
		void writeLinesToPLY(const std::string& filename, const std::vector<Line3D>& lines, const std::vector<int>& centerIDs);
		//Line3D findLineByID(int targetID)const;
		std::vector<Line3D> outLines();
		std::vector<int>outID();
		void getSimilarrityMatrix(std::vector<std::vector<double>>);
		void convertToSparseMatrixPlus();
		void computeEigenPlus();
		void computeEigenPlusPlus();
		void computeEigenRandomized();
		void computeEigenShift();
		void calculateNormalizedLaplacianMatrix();
		void normalizeEigenPairs();//标准化特征向量
		void storeEigenPairs();//存储到特征矩阵中
	};
	std::vector<Line3D> readLinesFromPLY(const std::string& filename);
	bool readPointFromPLY(const std::string& filename, std::vector<Vertex>& vertices);
	std::vector<std::vector<double>> spectralClustringComoleteFlowScheme(std::vector<Line3D>);
}