#include "Metrics.h"
#include <cmath>
#include "FLAB.h"

using namespace Coeus;

Metrics::Metrics()
{
}


Metrics::~Metrics()
{
}

double Metrics::euclidean_distance(const int p_x1, const int p_y1, const int p_x2, const int p_y2) {
	return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double Metrics::gaussian_distance(const double p_d, const double p_sigma) {
	return exp(-0.5 * pow(p_d / p_sigma, 2)) / (p_sigma * sqrt2PI);
}