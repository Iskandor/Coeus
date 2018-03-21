#include "LSOM_learning.h"
#include "LSOM_params.h"

using namespace Coeus;

LSOM_learning::LSOM_learning(LSOM* p_som, LSOM_params* p_params, SOM_analyzer* p_som_analyzer) : Base_SOM_learning(p_som, p_params, p_som_analyzer)
{
	_lsom = p_som;

	const int dim_input = p_som->get_input_group()->get_dim();
	const int dim_lattice = p_som->get_lattice()->get_dim();

	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_delta_lw = Tensor::Zero({ dim_lattice, dim_lattice });
}


LSOM_learning::~LSOM_learning()
{
}

void LSOM_learning::train(Tensor * p_input)
{
	const int winner = _lsom->find_winner(p_input);
	const int dim_input = _lsom->get_input_group()->get_dim();
	const int dim_lattice = _lsom->get_lattice()->get_dim();
	Tensor* oi = _lsom->get_output();
	Tensor* in = _lsom->get_input_group()->getOutput();
	Tensor* wi = _lsom->get_input_lattice()->get_weights();
	Tensor* li = _lsom->get_lattice_lattice()->get_weights();

	double theta = 0;
	const double alpha = static_cast<LSOM_params*>(_params)->alpha();
	const double beta = static_cast<LSOM_params*>(_params)->beta();

	_som_analyzer->update(_lsom, winner);

	for (int i = 0; i < dim_lattice; i++) {
		theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * alpha * (in->at(j) - wi->at(i, j)));
		}
		for (int j = 0; j < dim_lattice; j++) {
			_delta_lw.set(i, j, beta * (oi->at(j) * oi->at(i) - pow(oi->at(i), 2) * li->at(i, j)));
		}
	}

	_lsom->get_input_lattice()->update_weights(_delta_w);
	_lsom->get_lattice_lattice()->update_weights(_delta_lw);
}