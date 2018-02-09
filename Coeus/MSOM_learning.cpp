#include "MSOM_learning.h"

using namespace Coeus;

MSOM_learning::MSOM_learning(MSOM* p_msom, MSOM_params *p_params, SOM_analyzer* p_analyzer): Base_SOM_learning(p_msom, p_params, p_analyzer) {
	_msom = p_msom;

	const int dim_input = _msom->get_input_group()->getDim();
	const int dim_context = _msom->get_context_group()->getDim();
	const int dim_lattice = _msom->get_lattice()->getDim();

	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_delta_c = Tensor::Zero({ dim_lattice, dim_context });	

	_batch_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_batch_delta_c = Tensor::Zero({ dim_lattice, dim_context });

}

MSOM_learning::~MSOM_learning()
{
}

void MSOM_learning::init_msom(MSOM * p_source) const {
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

	_som_analyzer->update(_msom, winner);

	const double gamma1 = static_cast<MSOM_params*>(_params)->gamma1();
	const double gamma2 = static_cast<MSOM_params*>(_params)->gamma2();

	for (int i = 0; i < dim_lattice; i++) {
		const double theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * gamma1 * (in->at(j) - wi->at(i, j)));
			_delta_c.set(i, j, theta * gamma2 * (ct->at(j) - ci->at(i, j)));
		}
	}

	_msom->get_input_lattice()->update_weights(_delta_w);
	_msom->get_context_lattice()->update_weights(_delta_c);

	_batch_delta_w += _delta_w;
	_batch_delta_c += _delta_c;
}

void MSOM_learning::merge(vector<MSOM_learning*>& p_learners) {
	int size = p_learners.size();

	for(auto it = p_learners.begin(); it != p_learners.end(); ++it) {
		_delta_w += (*it)->_batch_delta_w;
		_delta_c += (*it)->_batch_delta_c;

		(*it)->_batch_delta_w.fill(0);
		(*it)->_batch_delta_c.fill(0);
	}

	_delta_w /= size;
	_delta_c /= size;

	_msom->get_input_lattice()->update_weights(_delta_w);
	_msom->get_context_lattice()->update_weights(_delta_c);

	_delta_w.fill(0);
	_delta_c.fill(0);
}

void MSOM_learning::reset_context() const {
	_msom->reset_context();
}
