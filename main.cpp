#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include<opencv2/opencv.hpp>
#include"Cluster.h"
int main(int argc, char* argv[]) 
{
	SpectralClustring::PointSimilarity similarityMatrix;
	similarityMatrix.getLine(SpectralClustring::readLinesFromPLY(argv[0]));
	similarityMatrix.outPutSimilarity();
	SpectralClustring::LaplacianMatrix laplaciacianMatrix;

}
