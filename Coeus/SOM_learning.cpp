#include "SOM_learning.h"
#include <iostream>

using namespace Coeus;


SOM_learning::SOM_learning(SOM* p_som) : Base_SOM_learning(p_som), _alpha0(0), _alpha(0) {
	_som = p_som;
}

SOM_learning::~SOM_learning()
{	
}

void SOM_learning::init_training(const double p_alpha, const double p_epochs) {
	Base_SOM_learning::init_training(p_epochs);
	_alpha0 = _alpha = p_alpha;
	_alpha = _alpha0 * exp(-_iteration / _lambda);
}

void SOM_learning::train(Tensor* p_input) {
	const int winner = _som->find_winner(p_input);	
	const int dim_input = _som->get_input_group()->getDim();
	const int dim_lattice = _som->get_lattice()->getDim();
	Tensor* wi = _som->get_input_lattice()->get_weights();
	Tensor* in = _som->get_input_group()->getOutput();

	double theta = 0;

	_som_analyzer->update(winner);

	for (int i = 0; i < dim_lattice; i++) {
		theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * _alpha * (in->at(j) - wi->at(i, j)));
		}
	}

	_som->get_input_lattice()->update_weights(_delta_w);
}

void SOM_learning::param_decay() {
	_som_analyzer->end_epoch();
	Base_SOM_learning::param_decay();
	_alpha = _alpha0 * exp(-_iteration / _lambda);
}


