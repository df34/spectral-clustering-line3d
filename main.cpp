#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include<opencv2/opencv.hpp>
#include"Cluster.h"
using namespace SpectralClustring;
int main(int argc, char* argv[]) 
{
	SpectralClustring::PointSimilarity similarityMatrix;
	//std::vector<Vertex>vertices;
	//readPointFromPLY(argv[1],vertices);
	similarityMatrix.getLine(SpectralClustring::readLinesFromPLY(argv[1]));
	SpectralClustring::spectralClustringComoleteFlowScheme(similarityMatrix.outLine());
	SpectralClustring::LaplacianMatrix laplaciacianMatrix;
	laplaciacianMatrix.getSimilarrityMatrix(similarityMatrix.outPutSimilarity());
	laplaciacianMatrix.calcuCluster();
	string filename = "E:\\谱聚类\\afterPly.ply";
	laplaciacianMatrix.writeLinesToPLY(filename, laplaciacianMatrix.outLines(), laplaciacianMatrix.outID());;

}
