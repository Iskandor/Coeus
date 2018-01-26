#include "MSOM_learning.h"

using namespace Coeus;

MSOM_learning::MSOM_learning(MSOM* p_msom, MSOM_params *p_params, SOM_analyzer* p_analyzer): Base_SOM_learning(p_msom, p_params, p_analyzer) {
	_msom = p_msom;

	const int dim_context = _msom->get_context_group()->getDim();
	const int dim_lattice = _msom->get_lattice()->getDim();
	_delta_c = Tensor::Zero({ dim_lattice, dim_context });
}

MSOM_learning::~MSOM_learning()
{
}

void Coeus::MSOM_learning::init_msom(MSOM * p_source)
{
	_msom->override_params(p_source);
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

	double gamma1 = static_cast<MSOM_params*>(_params)->gamma1();
	double gamma2 = static_cast<MSOM_params*>(_params)->gamma2();

	for (int i = 0; i < dim_lattice; i++) {
		const double theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * gamma1 * (in->at(j) - wi->at(i, j)));
			_delta_c.set(i, j, theta * gamma2 * (ct->at(j) - ci->at(i, j)));
		}
	}

	_msom->get_input_lattice()->update_weights(_delta_w);
	_msom->get_context_lattice()->update_weights(_delta_c);
}
