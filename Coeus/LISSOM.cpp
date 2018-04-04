#include "LISSOM.h"
#include "ActivationFunctions.h"

using namespace Coeus;

LISSOM::LISSOM(const string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const NeuralGroup::ACTIVATION p_activation, double p_gamma_e, double p_gamma_i) : SOM(p_id, p_input_dim, p_dim_x, p_dim_y, p_activation)
{
	_lateral_e = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lateral_e->init(Connection::UNIFORM, 0.01);
	_lateral_i = new Connection(p_dim_x * p_dim_y, p_dim_x * p_dim_y, "lattice", "lattice");
	_lateral_i->init(Connection::UNIFORM, 0.01);

	_auxoutput = Tensor::Zero({ _dim_x * _dim_y });
	_prime_activity = Tensor::Zero({ _dim_x * _dim_y });

	_gamma_e = p_gamma_e;
	_gamma_i = p_gamma_i;
}


LISSOM::~LISSOM()
{
}

void LISSOM::activate(Tensor* p_input) {
	_input_group->set_output(p_input);

	_prime_activity = *_afferent->get_weights() * *p_input;

	switch (_output_group->getActivationFunction()) {
		case NeuralGroup::LINEAR:
		_dist = _prime_activity.apply(ActivationFunctions::linear);
		break;
		case NeuralGroup::SIGMOID:
		_dist = _prime_activity.apply(ActivationFunctions::sigmoid);
		break;
		case NeuralGroup::TANH:
		_dist = _prime_activity.apply(ActivationFunctions::tanh);
		break;
		case NeuralGroup::RELU:
		_dist = _prime_activity.apply(ActivationFunctions::relu);
		break;
		default:
		break;
	}

	Tensor* lateral_e = _lateral_e->get_weights();
	Tensor* lateral_i = _lateral_i->get_weights();

	_auxoutput = _dist;

	for(int s = 0; s < 10; s++) {
		Tensor exc = *lateral_e * _auxoutput;
		Tensor inh = *lateral_i * _auxoutput;

		_auxoutput = _prime_activity + exc * _gamma_e - inh * _gamma_i;
		_auxoutput = _auxoutput.apply(ActivationFunctions::sigmoid);
	}

	_output_group->set_output(&_auxoutput);
}