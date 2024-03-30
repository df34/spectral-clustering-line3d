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
	SpectralClustring::LaplacianMatrix laplaciacianMatrix;
	//std::vector<Vertex>vertices;
	//readPointFromPLY(argv[1],vertices);
	similarityMatrix.getLine(SpectralClustring::readLinesFromPLY(argv[1]));
	
	
	laplaciacianMatrix.getSimilarrityMatrix(SpectralClustring::spectralClustringComoleteFlowScheme(similarityMatrix.outLine()));
	laplaciacianMatrix.calcuCluster();
	string filename = "E:\\谱聚类\\afterPly.ply";
	laplaciacianMatrix.writeLinesToPLY(filename, laplaciacianMatrix.outLines(), laplaciacianMatrix.outID());;

}
