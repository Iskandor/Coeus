#include "Metrics.h"
#include <cmath>
#include "Coeus.h"

using namespace Coeus;

Metrics::Metrics()
{
}


Metrics::~Metrics()
{
}

float Metrics::euclidean_distance(const int p_x1, const int p_y1, const int p_x2, const int p_y2) {
	return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

float Metrics::gaussian_distance(const float p_d, const float p_sigma, const float p_mu) {
	return  exp(-pow(p_d - p_mu, 2) / (2 * pow(p_sigma, 2))) / (p_sigma * sqrt2PI);
}

float Metrics::binary_distance(const float p_d, const float p_h) {
	return p_d < p_h ? 1 : -1;
}