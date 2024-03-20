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
typedef struct Cluster3
{
public:
	double m_x;
	double m_y;
	double m_z;
	//与三条基准线沿法向量的距离
	double m_dis1;
	double m_dis2;
	double m_dis3;
	Cluster3() {
		m_x = 0;
		m_y = 0;
		m_z = 0;
		m_dis1 = 0;
		m_dis2 = 0;
		m_dis3 = 0;
	};

	Cluster3(double m_x, double m_y, double m_z, double dis1, double dis2, double dis3) :m_x(m_x), m_y(m_y), m_z(m_z), m_dis1(dis1), m_dis2(dis2), m_dis3(dis3) {};
	Cluster3(double m_x, double m_y, double m_z) :m_x(m_x), m_y(m_y), m_z(m_z) {};
};

typedef struct Line3D
{
	double X1;
	double Y1;
	double Z1;							//线特征起点三维坐标
	double X2;
	double Y2;
	double Z2;							//线特征终点三维坐标
	//double avgZ;
	Line3D() : X1(0), Y1(0), Z1(0), X2(0), Y2(0), Z2(0) {}
	Line3D(double X1, double Y1, double Z1, double X2, double Y2, double Z2) :X1(X1), Y1(Y1), Z1(Z1), X2(X2), Y2(Y2), Z2(Z2) {};
};

typedef struct Vector6D
{
	double xc;
	double yc;
	double zc;//中点坐标
	double u;
	double v;
	double w;//方向向量
	double x;
	double y;
	double z;//终点坐标

	int idImage;//这条线的来源影像
	int idLine2D;//这条线对应的2D线的id
	Vector6D()
	{
		xc = 0;
		yc = 0;
		zc = 0;
		u = 0;
		v = 0;
		w = 0;
		x = 0;
		y = 0;
		z = 0;
	};
	Vector6D(double xc, double yc, double zc, double u, double v, double w, double x, double y, double z) :xc(xc), yc(yc), zc(zc), u(u), v(v), w(w), x(x), y(y), z(z) {};
	//中点坐标和方向向量

};

typedef struct Spherical
{
	double angle1;
	double angle2;
	double dis2origin;
};

typedef vector<std::vector<Vector6D>> LineCluster;	//聚类好的线特征
typedef vector<std::vector<Line3D>> EdgePoint;		//存储线特征两端点
typedef vector<Spherical>SphericalPoint;			//球形坐标系下的三维点

typedef struct
{
	double Coord[3];				//点坐标 x y z
	float Normal[3];				//点的法向量 nx ny nz
	unsigned char RGB[3];			//点的颜色 red green blue
	float value;					//点密度  value
}PLYPoint;

typedef struct
{
	int vertexID[3];				//三角面顶点索引
}PLYFacet;

struct PLYOptions
{
	bool isWithNormal;				//是否带有法向量
	bool isWithRGB;					//是否带有颜色
	bool isWithValue;				//是否带有附带值
	bool isASCII;					//是否采用二进制
	PLYOptions() : isWithNormal(false), isWithRGB(false), isWithValue(false), isASCII(false) {};
};

//class Cluster
//{
//	
//private:
//	void vector_angle(Vector6D each_vector, double& theta_1, double& theta_2);
//	void kmeansCluster(vector<Cluster3>& dataArr, vector<vector<int>>& clusterIndex, double thres);
//	void findThreeBaseLines(std::vector<Vector6D> every_linearr, Vector6D& L1, Vector6D& L2, Vector6D& L3);
//	double GetEuclideanDistance1(Vector6D vector6d1, Vector6D vector6d2);/*计算6维向量的欧式距离（点到三维直线的距离）*/
//	void kmeansCluster2(std::vector<Cluster3>& dataArr, std::vector<std::vector<int>>& clusterIndex, double thres);
//	void RegionGrowing(std::vector<Line3D>& oneCluster);
//	void RegionGrowing2(std::vector<Line3D>& oneCluster, std::vector<Line3D>& oneCluster2);
//	bool isPointOnTheLine(Line3D line, cv::Point3d point);
//	void Line3D2Vec6D(std::vector<Line3D> oneCluster, std::vector<Vector6D>& oneLineCluster);
//	void findFarthestLine(std::vector<Vector6D> every_linearr, Vector6D& L, Vector6D& L2);//查找距离L1最远的线L2
//	double GetEuclideanDistance2(Point3d P0, Point3d P1);
//	void kmeansCluster3(std::vector<Cluster3>& dataArr, std::vector<std::vector<int>>& clusterIndex, double thres);
//	bool NoLineConnect(std::vector<Line3D>& edgepointcluster);//判断是否是断裂的类
//	
//public:
//	bool readPLYFile(const char* path, std::vector<PLYPoint>& meshPointArr, std::vector<PLYFacet>& meshFaectsArr, PLYOptions& options);
//	void readPLYFile(string path, std::vector<Spherical>& clusterPoint);
//public:
//	
//	//void GenerateLineandPoint(LineCluster& lineCluster, EdgePoint& edgePointCluster, std::vector<Spherical>& clusterPoints);
//	void LineClusteringByAvgZ(std::vector<Vector6D>& lineArr, LineCluster& lineCluster, std::vector<Line3D>& endpointarr, EdgePoint& edgepointCluster);
//	//方向向量角度聚类
//	void LineClusteringAngle(std::vector<Vector6D>& lineArr, LineCluster& lineCluster, std::vector<Line3D>& endpointarr, EdgePoint& edgepointCluster, SphericalPoint& sphericalPoint, std::vector<Spherical>& clusterPoints);
//	// 距离聚类
//	void LineClusteringDistance(LineCluster& lineCluster, EdgePoint& edgepointcluster);//距离聚类
//	//距离聚类(重合但不相交的直线）
//	void LineClusteringDistance2(LineCluster& lineCluster, EdgePoint& edgepointcluster,SphericalPoint& sphericalPoint);
//	//连通性聚类
//	void LineClusteringConnectivity(LineCluster& lineCluster, EdgePoint& edgepointCluster);//连通性聚类
//	void LineClusteringConnectivity2(LineCluster& lineCluster, EdgePoint& edgepointCluster);//连通性聚类
//	//长度聚类
//	void LineClusteringLength(LineCluster& lineCluster, EdgePoint& edgepointCluster);
//	//聚类过滤器，过滤只有一条直线的直线簇
//	void LineClusteringFilter(LineCluster& lineCluster, EdgePoint& edgepointcluster);
//	//聚类过滤器，过滤断裂的类
//	void LineClusteringFilter2(LineCluster& lineCluster, EdgePoint& edgepointcluster);
//
//	//投影聚类
//	void LineClusteringProjection(LineCluster& lineCluster, EdgePoint& edgepointcluster);
//	void GetEuclideanDistance3(std::vector<Vector6D>& lineArr, SphericalPoint& sphericalPoint);
//	void OutputClusterLine(const char* outputFile, EdgePoint endpoints);//输出聚类结果
//	void OutputPointCloud(const char* outputFile, SphericalPoint sphericalPoint);//输出聚类结果
//};
namespace SpectralClustring
{
	class PointSimilarity
	{
	private:
		std::vector<Line3D>lines;
		static double linepearsonCorrelation(Line3D line1, Line3D line2);
		static double angleGap(Line3D line1, Line3D line2);
	public:
		void assignmentSimilarity();
	};
	typedef struct LinearFeature
	{
		double x_midpoint;
		double y_midpoint;
		double z_midpoint;
		double latitude;
		double longitude;
	};
}