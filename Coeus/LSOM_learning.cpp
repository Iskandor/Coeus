#include "LSOM_learning.h"
#include "LSOM_params.h"
#include "Metrics.h"

using namespace Coeus;

LSOM_learning::LSOM_learning(LSOM* p_som, LSOM_params* p_params, SOM_analyzer* p_som_analyzer) : Base_SOM_learning(p_som, p_params, p_som_analyzer)
{
	_past = 100;
	_lsom = p_som;

	const int dim_input = p_som->get_input_group()->get_dim();
	const int dim_lattice = p_som->get_lattice()->get_dim();

	_delta_b = Tensor::Zero({ dim_lattice });
	_delta_w = Tensor::Zero({ dim_lattice, dim_input });
	_delta_lw = Tensor::Zero({ dim_lattice, dim_lattice });

	_avg = Tensor::Zero({ dim_lattice });
	_mean = Tensor::Zero({ dim_lattice });
	_deviation = Tensor::Zero({ dim_lattice });

	_s = 0;
	_t = 0;

	_lsom->get_afferent()->normalize_weights(Connection::L1_NORM);
	_lsom->get_lateral()->normalize_weights(Connection::L1_NORM);
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
	Tensor* in = _lsom->get_input_group()->get_output();
	Tensor* wi = _lsom->get_afferent()->get_weights();
	Tensor* li = _lsom->get_lateral()->get_weights();
	Tensor* bi = _lsom->get_lattice()->get_bias();

	const double alpha = dynamic_cast<LSOM_params*>(_params)->alpha();
	const double beta = dynamic_cast<LSOM_params*>(_params)->beta();

	_som_analyzer->update(_lsom, winner);
	_winners.insert(winner);
	
	if (_hist.size() == _s) {
		_hist.push_back(Tensor::Zero({ _past, dim_lattice }));
	}

	for (int i = 0; i < dim_lattice; i++) {
		_hist[_s].set(_t, i, oi->at(i));

		_mean[i] = 0;
		for(int t = 0; t < _past; t++) {
			_mean[i] += _hist[_s].at(t, i);
		}

		_mean[i] /= _past;

		_deviation[i] = 0;
		for (int t = 0; t < _past; t++) {
			_deviation[i] += pow(_hist[_s].at(t, i) - _mean[i], 2);
		}
		_deviation[i] /= _past - 1;
	}

	//_delta_w.fill(0);

	for (int i = 0; i < dim_lattice; i++) {

		for (int j = 0; j < dim_input; j++) {
			_delta_w.set(i, j, alpha * (in->at(j) * oi->at(i)));
		}

		for (int j = 0; j < dim_lattice; j++) {
			double cov = 0;

			for (int t = 0; t < _past; t++) {
				cov += (_hist[_s].at(t, i) - _mean[i]) * (_hist[_s].at(t, j) - _mean[j]);
			}

			cov /= _past;

			const double ro = cov / (sqrt(_deviation[i]) * sqrt(_deviation[j]));

			//cout << winner << " " << j << " " << ro << endl;

			if ( ro < 0) {
				int s = 0;
			}

			const double l = _dist_matrix.at(i, j) == 0 ? 0 : 1 / _dist_matrix.at(i, j);
			const double val = beta * ro * l * abs(oi->at(i) * oi->at(j));
			_delta_lw.set(i, j, val);

			if (_delta_lw.at(j,i) != _delta_lw.at(j, i)) {
				int s = 0;
			}
		}

		double v = oi->at(i) - _avg[i];

		_delta_b[i] = alpha * -v * abs(oi->at(i));

		if (v < 0) {
			v = 0;
		}

		_avg[i] = _avg[i] * (_s / (_s + 1)) + oi->at(i) / (_s + 1);

	}

	_s++;

	_lsom->get_output_group()->update_bias(_delta_b);

	_lsom->get_afferent()->update_weights(_delta_w);
	_lsom->get_lateral()->update_weights(_delta_lw);

	_lsom->get_afferent()->normalize_weights(Connection::L1_NORM);
	_lsom->get_lateral()->normalize_weights(Connection::L1_NORM);


	if (wi->at(0) != wi->at(0)) {
		int i = 0;
	}

	if (li->at(0) != li->at(0)) {
		int i = 0;
	}
}

void LSOM_learning::update() {
	_t++;
	if (_t == _past) _t = 0;
	_s = 0;
	_avg.fill(0);
}
