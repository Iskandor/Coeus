#include "LSOM.h"
#include "ActivationFunctions.h"
#include "Metrics.h"

using namespace Coeus;

LSOM::LSOM(const string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation)
{
	_type = TYPE::LSOM;
	_lattice_lattice = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lattice_lattice->init(Connection::UNIFORM, 0.01);

	_auxoutput = Tensor::Zero({ _dim_x * _dim_y });
}


LSOM::~LSOM()
{
	delete _lattice_lattice;
}

void LSOM::activate(Tensor * p_input)
{
	_input_group->set_output(p_input);
	calc_distance();

	switch (_output_group->getActivationFunction()) {
	case NeuralGroup::LINEAR:
		_dist = _dist.apply(ActivationFunctions::linear);
		break;
	case NeuralGroup::EXPONENTIAL:
		_dist = _dist.apply(ActivationFunctions::exponential);
		break;
	case NeuralGroup::KEXPONENTIAL:
		_dist = _dist.apply(ActivationFunctions::kexponential);
		break;
	case NeuralGroup::GAUSS:
		_dist = _dist.apply(ActivationFunctions::gauss);
		break;
	default:
		break;
	}

	Tensor* lateral_w = _lattice_lattice->get_weights();

	int x1 = 0;
	int y1 = 0;
	int x2 = 0;
	int y2 = 0;

	for (int i = 0; i < _dim_x * _dim_y; i++) {
		get_position(i, x1, y1);
		for (int n = 0; n < _dim_x * _dim_y; n++) {
			get_position(n, x2, y2);
			
			//double d = Metrics::euclidean_distance(x1, y1, x2, y2);

			_auxoutput.set(i, _auxoutput.at(i) + _dist.at(n) * lateral_w->at(i, n)); // * (1.0 / (d == 0 ? 1 : d))
		}
	}

	_output_group->set_output(&_auxoutput);
	_auxoutput.fill(0);
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
