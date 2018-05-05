#include "LSOM.h"
#include "ActivationFunctions.h"
#include "Metrics.h"

using namespace Coeus;

LSOM::LSOM(const string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation)
{
	_type = TYPE::LSOM;
	_lateral = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lateral->init(Connection::GLOROT_UNIFORM);

	_auxoutput = Tensor::Zero({ _dim_x * _dim_y });
}


LSOM::~LSOM()
{
	delete _lateral;
}

void LSOM::activate(Tensor * p_input)
{
	_input_group->set_output(p_input);


	calc_distance();

	switch (_output_group->get_activation_function()) {
	case NeuralGroup::LINEAR:
		_auxoutput = Tensor::apply(_dist, ActivationFunctions::linear);
		break;
	case NeuralGroup::EXPONENTIAL:
		_auxoutput = Tensor::apply(_dist, ActivationFunctions::exponential);
		break;
	case NeuralGroup::KEXPONENTIAL:
		_auxoutput = Tensor::apply(_dist, ActivationFunctions::kexponential);
		break;
	case NeuralGroup::GAUSS:
		_auxoutput = Tensor::apply(_dist, ActivationFunctions::gauss);
		break;
	default:
		break;
	}

	Tensor* lateral_w = _lateral->get_weights();

	for (int i = 0; i < _dim_x * _dim_y; i++) {
		double w = 0;
		for (int n = 0; n < _dim_x * _dim_y; n++) {
				w += _auxoutput.at(n) * lateral_w->at(i, n);				
		}
		_auxoutput.inc(i, w);
	}
	_auxoutput = Tensor::apply(_auxoutput, ActivationFunctions::relu);

	_output_group->set_output(&_auxoutput);
}

int LSOM::find_winner(Tensor * p_input)
{
	activate(p_input);
	
	_winner = 0;

	for (int i = 0; i < _output_group->get_dim(); i++) {
		if (_output_group->get_output()->at(_winner) < _output_group->get_output()->at(i)) {
			_winner = i;
		}
	}

	return _winner;
}
