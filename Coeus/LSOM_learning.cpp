#include "LSOM_learning.h"

using namespace Coeus;

LSOM_learning::LSOM_learning(LSOM* p_som, SOM_params* p_params, SOM_analyzer* p_som_analyzer) : Base_SOM_learning(p_som, p_params, p_som_analyzer)
{
	_lsom = p_som;
}


LSOM_learning::~LSOM_learning()
{
}

void LSOM_learning::train(Tensor * p_input)
{
	const int winner = _lsom->find_winner(p_input);
	const int dim_input = _lsom->get_input_group()->get_dim();
	const int dim_lattice = _lsom->get_lattice()->get_dim();
	Tensor* wi = _lsom->get_input_lattice()->get_weights();
	Tensor* in = _lsom->get_input_group()->getOutput();

	double theta = 0;
	const double alpha = static_cast<SOM_params*>(_params)->alpha();

	_som_analyzer->update(_lsom, winner);

	for (int i = 0; i < dim_lattice; i++) {
		theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);
		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * alpha * (in->at(j) - wi->at(i, j)));
		}
		for (int j = 0; j < dim_lattice; j++) {
		}
	}

	_lsom->get_input_lattice()->update_weights(_delta_w);
}
