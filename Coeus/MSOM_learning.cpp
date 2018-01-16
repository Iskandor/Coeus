#include "MSOM_learning.h"

using namespace Coeus;

MSOM_learning::MSOM_learning(MSOM* p_msom): Base_SOM_learning(p_msom), _gamma1_0(0), _gamma1(0), _gamma2_0(0), _gamma2(0) {
	_msom = p_msom;

	const int dim_context = _msom->get_context_group()->getDim();
	const int dim_lattice = _msom->get_lattice()->getDim();
	_delta_c = Tensor::Zero({ dim_lattice, dim_context });
}

MSOM_learning::~MSOM_learning()
{
}

void MSOM_learning::init_training(const double p_gamma1, const double p_gamma2, const double p_epochs) {
	Base_SOM_learning::init_training(p_epochs);
	_gamma1_0 = p_gamma1;
	_gamma2_0 = p_gamma2;
	_gamma1 = _gamma1_0 * exp(-_iteration / _lambda);
	_gamma2 = _gamma2_0 * exp(-_iteration / _lambda);
}

void MSOM_learning::train(Tensor* p_input) {
	const int winner = _msom->find_winner(p_input);
	const int dim_input = _msom->get_input_group()->getDim();
	const int dim_lattice = _msom->get_lattice()->getDim();

	_msom->update_context();

	Tensor* wi = _msom->get_input_lattice()->get_weights();
	Tensor* ci = _msom->get_context_lattice()->get_weights();
	Tensor* in = _msom->get_input_group()->getOutput();
	Tensor* ct = _msom->get_context_group()->getOutput();

	_som_analyzer->update(winner);

	for (int i = 0; i < dim_lattice; i++) {
		const double theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * _gamma1 * (in->at(j) - wi->at(i, j)));
			_delta_c.set(i, j, theta * _gamma2 * (ct->at(j) - ci->at(i, j)));
		}
	}

	_msom->get_input_lattice()->update_weights(_delta_w);
	_msom->get_context_lattice()->update_weights(_delta_c);
}

void MSOM_learning::param_decay() {
	_som_analyzer->end_epoch();
	Base_SOM_learning::param_decay();
	_gamma1 = _gamma1_0 * exp(-_iteration / _lambda);
	_gamma2 = _gamma2_0 * exp(-_iteration / _lambda);
}
