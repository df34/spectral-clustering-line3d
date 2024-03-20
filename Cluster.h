#pragma once
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
//#include "rply.h"
#include<fstream>
using namespace std;
using namespace cv;
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
	int serialNumber;
	//double avgZ;
	Line3D() : X1(0), Y1(0), Z1(0), X2(0), Y2(0), Z2(0) ,serialNumber(0){}
	Line3D(double X1, double Y1, double Z1, double X2, double Y2, double Z2) :X1(X1), Y1(Y1), Z1(Z1), X2(X2), Y2(Y2), Z2(Z2) ,serialNumber(serialNumber){};
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
		double assignmentSimilarity();
		double getSimilarity(int row, int col) const;
	};
	typedef struct LinearFeature
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

	class LaplacianMatrix
	{
	private:
		std::vector<std::vector<double>> metricMatrix;
		std::vector<std::vector<double>> LaplacianMatrix;
		std::vector<std::vector<double>> similarityMatrix;

	public:
		void calculateMetricMatrix(std::vector<std::vector<double>>& metricMatrix);
	};
}