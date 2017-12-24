#include "SOM_learning.h"
#include "FLAB.h"
#include <algorithm>
#include <iostream>

using namespace Coeus;


SOM_learning::SOM_learning(SOM* p_som) {
	_som = p_som;
	_som_analyzer = new SOM_analyzer(p_som);

	_sigma0 = sqrt(max(_som->dim_x(), _som->dim_y()));
	
	_lambda = 1;

	const int dim_input = _som->get_input_group()->getDim();
	const int dim_lattice = _som->get_lattice()->getDim();
	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
}

SOM_learning::~SOM_learning()
{
	delete _som_analyzer;
}

void SOM_learning::init_training(const double p_alpha, const double p_epochs) {
	_iteration = 0;
	_alpha0 = _alpha = p_alpha;
	_lambda = p_epochs / log(_sigma0);
	_sigma = _sigma0 * exp(-_iteration / _lambda);
	_alpha = _alpha0 * exp(-_iteration / _lambda);
}

void SOM_learning::train(Tensor* p_input) {
	const int winner = _som->find_winner(p_input);	
	const int dim_input = _som->get_input_group()->getDim();
	const int dim_lattice = _som->get_lattice()->getDim();
	Tensor* wi = _som->get_lattice_connection()->get_weights();
	Tensor* in = _som->get_input_group()->getOutput();

	double theta = 0;

	int x1;
	int y1;
	int x2;
	int y2;

	_som_analyzer->update(winner);
	_som->get_position(winner, x1, y1);
	_delta_w.fill(0);


	for (int i = 0; i < dim_lattice; i++) {
		_som->get_position(i, x2, y2);
		theta = calc_neighborhood(x1, y1, x2, y2, GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * _alpha * (in->at(j) - wi->at(i, j)));
		}
	}

	_som->get_lattice_connection()->update_weights(_delta_w);
}

void SOM_learning::param_decay() {
	_som_analyzer->end_epoch();
	_iteration++;
	_sigma = _sigma0 * exp(-_iteration / _lambda);
	_alpha = _alpha0 * exp(-_iteration / _lambda);
}

double SOM_learning::calc_neighborhood(const int p_x1, const int p_y1, const int p_x2, const int p_y2, const NEIGHBORHOOD_TYPE p_type) const {
	double result = 0;

	switch (p_type) {
	case EUCLIDEAN:
		result = 1.0 / euclidean_distance(p_x1, p_y1, p_x2, p_y2);
		break;
	case GAUSSIAN:
		result = gaussian_distance(euclidean_distance(p_x1, p_y1, p_x2, p_y2), _sigma);
		break;
	}

	return result;
}

double SOM_learning::euclidean_distance(const int p_x1, const int p_y1, const int p_x2, const int p_y2) const {
	return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double SOM_learning::gaussian_distance(const double p_d, const double p_sigma) const {
	return exp(-0.5 * pow(p_d / p_sigma, 2)) / (p_sigma * sqrt2PI);
}
