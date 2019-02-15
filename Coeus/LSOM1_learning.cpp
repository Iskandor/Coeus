#include "LSOM1_learning.h"
#include "LSOM1_params.h"
#include "Metrics.h"

using namespace Coeus;

LSOM1_learning::LSOM1_learning(LSOM1* p_som, LSOM1_params* p_params, SOM_analyzer* p_som_analyzer) : Base_SOM_learning(p_som, p_params, p_som_analyzer)
{
	_lsom = p_som;

	const int dim_input = p_som->get_input_group<SimpleCellGroup>()->get_dim();
	const int dim_lattice = p_som->get_lattice()->get_dim();

	_friendship = Tensor::Value({ dim_lattice }, 1);
	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_delta_lw = Tensor::Zero({ dim_lattice, dim_lattice });
}


LSOM1_learning::~LSOM1_learning()
{
}

void LSOM1_learning::train(Tensor * p_input)
{
	const int winner = _lsom->find_winner(p_input);
	const int dim_input = _lsom->get_input_group<SimpleCellGroup>()->get_dim();
	const int dim_lattice = _lsom->get_lattice()->get_dim();
	Tensor* oi = _lsom->get_output();
	Tensor* in = _lsom->get_input_group<SimpleCellGroup>()->get_output();
	Tensor* wi = _lsom->get_afferent()->get_weights();
	Tensor* li = _lsom->get_lateral()->get_weights();

	float theta = 0;
	const float alpha = static_cast<LSOM1_params*>(_params)->alpha();
	const float beta = static_cast<LSOM1_params*>(_params)->beta();

	_som_analyzer->update(_lsom, winner);
	_winners.insert(winner);

	Tensor norm = Tensor::Zero({ dim_lattice });

	for (int i = 0; i < dim_lattice; i++) {
		theta = calc_neighborhood(_dist_matrix.at(winner, i), GAUSSIAN);

		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, theta * alpha * (in->at(j) - wi->at(i, j)));
		}

		for (int j = 0; j < dim_lattice; j++) {
			const float lambda = -_dist_matrix.at(i, j) + pow(_friendship.at(i), 4);
			//const float lambda = Metrics::binary_distance(_dist_matrix.at(i, j), _friendship.at(i));
			//const float lambda = Metrics::gaussian_distance(_dist_matrix.at(i, j), _friendship.at(i)) - 0.25;
			
			const float val = lambda * beta * (oi->at(j) * oi->at(i) - pow(oi->at(i), 2) * abs(li->at(i, j)));
			_delta_lw.set(i, j, val);
			norm.inc(i, abs(li->at(i, j) + val));
		}
	}

	_lsom->get_afferent()->update_weights(_delta_w);
	_lsom->get_lateral()->update_weights(_delta_lw);

	for (int i = 0; i < dim_lattice; i++) {
		for (int j = 0; j < dim_lattice; j++) {
			li->set(i, j, li->at(i, j) / norm[i]);
		}
	}
}

void LSOM1_learning::update_friendship() {
	const int dim_lattice = _lsom->get_lattice()->get_dim();

	for (int i = 0; i < dim_lattice; i++) {
		if (_winners.count(i) > 0) {
			_friendship.set(i, _friendship.at(i) * 0.99);
			/*
			if (_friendship.at(i) < 2) {
				_friendship.set(i, 2);
			}
			*/
		}
		else {
			_friendship.set(i, _friendship.at(i) * 1.001);
		}
	}

	_winners.clear();
}
