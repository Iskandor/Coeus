#include "SOM_learning.h"
#include <iostream>

using namespace Coeus;


SOM_learning::SOM_learning(SOM* p_som, SOM_params* p_params, SOM_analyzer* p_analyzer) : Base_SOM_learning(p_som, p_params, p_analyzer) {
	const int dim_input = p_som->get_input_group()->get_dim();
	const int dim_lattice = p_som->get_lattice()->get_dim();

	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_batch_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_som = p_som;
}

SOM_learning::~SOM_learning()
{	
}

void SOM_learning::init_som(SOM * p_source) const {
	_som->override_params(p_source);
}

void SOM_learning::train(Tensor* p_input) {
	const int winner = _som->find_winner(p_input);	
	const int dim_input = _som->get_input_group()->get_dim();
	const int dim_lattice = _som->get_lattice()->get_dim();
	Tensor* wi = _som->get_afferent()->get_weights();
	Tensor* in = _som->get_input_group()->getOutput();

	double theta = 0;
	const double alpha = static_cast<SOM_params*>(_params)->alpha();

	_som_analyzer->update(_som, winner);

	for (int i = 0; i < dim_lattice; i++) {
		theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * alpha * (in->at(j) - wi->at(i, j)));
		}
	}

	_som->get_afferent()->update_weights(_delta_w);
}

void SOM_learning::merge(vector<SOM_learning*>& p_learners) {
	const int size = p_learners.size();

	for (auto it = p_learners.begin(); it != p_learners.end(); ++it) {
		_delta_w += (*it)->_batch_delta_w;

		(*it)->_batch_delta_w.fill(0);
	}

	_delta_w /= size;

	_som->get_afferent()->update_weights(_delta_w);

	_delta_w.fill(0);
}