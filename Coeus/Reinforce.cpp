#include "Reinforce.h"
#include "TensorOperator.h"

using namespace Coeus;

Reinforce::Reinforce(NeuralNetwork* p_network, const float p_alpha)
{
	_network = p_network;
	_alpha = p_alpha;

	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new ReinforceRule(_network_gradient, p_alpha);
}


Reinforce::~Reinforce()
{
	delete _update_rule;
	delete _network_gradient;
}

void Reinforce::train(Tensor* p_state, const int p_action, float p_delta)
{
	Tensor mask({ _network->get_output_dim() }, Tensor::ZERO);
	mask[p_action] = 1;

	_network->activate(p_state);

	const float pi_as = 1.f / _network->get_output()->at(p_action);

	_network_gradient->calc_gradient(&mask);
	map<string, Tensor> *gradient = _network_gradient->get_gradient();

	for (auto& it : *gradient)
	{
		TensorOperator::instance().vc_prod(it.second.arr(), pi_as, it.second.arr(), it.second.size());
	}

	_update_rule->calc_update();
}
