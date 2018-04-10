#include "LSOM.h"
#include "ActivationFunctions.h"
#include "Metrics.h"

using namespace Coeus;

LSOM::LSOM(const string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation)
{
	_type = TYPE::LSOM;
	_lateral = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lateral->init(Connection::UNIFORM, 0.01);

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

	switch (_output_group->getActivationFunction()) {
	case NeuralGroup::LINEAR:
		_auxoutput = _dist.apply(ActivationFunctions::linear);
		break;
	case NeuralGroup::EXPONENTIAL:
		_auxoutput = _dist.apply(ActivationFunctions::exponential);
		break;
	case NeuralGroup::KEXPONENTIAL:
		_auxoutput = _dist.apply(ActivationFunctions::kexponential);
		break;
	case NeuralGroup::GAUSS:
		_auxoutput = _dist.apply(ActivationFunctions::gauss);
		break;
	default:
		break;
	}

	Tensor* lateral_w = _lateral->get_weights();

	for(int s = 0; s < 10; s++) {
		for (int i = 0; i < _dim_x * _dim_y; i++) {
			double w = 0;
			for (int n = 0; n < _dim_x * _dim_y; n++) {
				 w += _auxoutput.at(n) * lateral_w->at(i, n);				
			}
			_auxoutput.set(i, _dist.at(i) + w);
		}
		//_auxoutput = _auxoutput.apply(ActivationFunctions::sigmoid);
	}

	_output_group->set_output(&_auxoutput);
}

int LSOM::find_winner(Tensor * p_input)
{
	activate(p_input);
	
	_winner = 0;

	for (int i = 0; i < _output_group->get_dim(); i++) {
		if (_output_group->getOutput()->at(_winner) < _output_group->getOutput()->at(i)) {
			_winner = i;
		}
	}

	return _winner;
}
