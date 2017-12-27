#include "Base_SOM_learning.h"
#include <algorithm>
#include "FLAB.h"

using namespace Coeus;

Base_SOM_learning::Base_SOM_learning(SOM* p_som) {
	_som_analyzer = new SOM_analyzer(p_som);

	_sigma0 = sqrt(max(p_som->dim_x(), p_som->dim_y()));

	_lambda = 1;

	const int dim_input = p_som->get_input_group()->getDim();
	const int dim_lattice = p_som->get_lattice()->getDim();
	_delta_w = Tensor::Zero({ dim_lattice, dim_input });

	_dist_matrix = Tensor::Zero({ dim_lattice, dim_lattice });

	int x1;
	int y1;
	int x2;
	int y2;

	for (int i = 0; i < dim_lattice; i++) {
		p_som->get_position(i, x1, y1);
		for (int j = 0; j < dim_lattice; j++) {
			p_som->get_position(j, x2, y2);
			_dist_matrix.set(i, j, euclidean_distance(x1, y1, x2, y2));
		}
	}

}


Base_SOM_learning::~Base_SOM_learning()
{
	delete _som_analyzer;
}

void Base_SOM_learning::param_decay() {
	_iteration++;
	_sigma = _sigma0 * exp(-_iteration / _lambda);
}

void Base_SOM_learning::init_training(const double p_epochs) {
	_iteration = 0;
	_lambda = p_epochs / log(_sigma0);
	_sigma = _sigma0 * exp(-_iteration / _lambda);
}

double Base_SOM_learning::calc_neighborhood(const double p_d, const NEIGHBORHOOD_TYPE p_type) const {
	double result = 0;

	switch (p_type) {
	case EUCLIDEAN:
		result = 1.0 / p_d;
		break;
	case GAUSSIAN:
		result = gaussian_distance(p_d, _sigma);
		break;
	}

	return result;
}

double Base_SOM_learning::euclidean_distance(const int p_x1, const int p_y1, const int p_x2, const int p_y2) const {
	return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double Base_SOM_learning::gaussian_distance(const double p_d, const double p_sigma) const {
	return exp(-0.5 * pow(p_d / p_sigma, 2)) / (p_sigma * sqrt2PI);
}