#define _USE_MATH_DEFINES
#include "metrics.h"
#include <cmath>

metrics::metrics()
= default;


metrics::~metrics()
= default;

/**
 * \brief Calculate Euclidean distance between two 2D points
 * \param p_x1 x coordinate of point1
 * \param p_y1 y coordinate of point1 
 * \param p_x2 x coordinate of point2
 * \param p_y2 y coordinate of point2
 * \return distance value
 */
float metrics::euclidean_distance(const int p_x1, const int p_y1, const int p_x2, const int p_y2) {
	return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

/**
 * \brief Calculate Gaussian distance between 1D points and Gaussian(mu,sigma)
 * \param p_d x coordinate of point
 * \param p_sigma standard deviation
 * \param p_mu mean value
 * \return distance value
 */
float metrics::gaussian_distance(const float p_d, const float p_sigma, const float p_mu) {
	return  exp(-pow(p_d - p_mu, 2) / (2 * pow(p_sigma, 2))) / (p_sigma * sqrt(2 * M_PI));
}

/**
 * \brief Calculate binary distance of 1D point and some center
 * \param p_d x coordinate of point
 * \param p_h coordinate of center
 * \return p_d < p_h then 1 else -1
 */
float metrics::binary_distance(const float p_d, const float p_h) {
	return p_d < p_h ? 1 : -1;
}