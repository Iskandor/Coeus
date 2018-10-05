#include "LSOM1.h"
#include "ActivationFunctions.h"
#include "Metrics.h"

using namespace Coeus;

LSOM1::LSOM1(const string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const ACTIVATION p_activation) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation)
{
	_type = TYPE::LSOM;
	_lateral = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lateral->init(Connection::GLOROT_UNIFORM);

	_auxoutput = Tensor::Zero({ _dim_x * _dim_y });
}


LSOM1::~LSOM1()
{
	delete _lateral;
}

void LSOM1::activate(Tensor * p_input)
{
	_input_group->set_output(p_input);

	calc_distance();

	_lattice_group->get_activation_function()->activate(_dist);

	Tensor* lateral_w = _lateral->get_weights();

	for (int i = 0; i < _dim_x * _dim_y; i++) {
		double w = 0;
		for (int n = 0; n < _dim_x * _dim_y; n++) {
				w += _auxoutput.at(n) * lateral_w->at(i, n);				
		}
		_auxoutput.inc(i, w);
	}
	_auxoutput = Tensor::apply(_auxoutput, ActivationFunctions::relu);

	_lattice_group->set_output(&_auxoutput);
}

int LSOM1::find_winner(Tensor * p_input)
{
	activate(p_input);
	
	_winner = 0;

	for (int i = 0; i < _lattice_group->get_dim(); i++) {
		if (_lattice_group->get_output()->at(_winner) < _lattice_group->get_output()->at(i)) {
			_winner = i;
		}
	}

	return _winner;
}
