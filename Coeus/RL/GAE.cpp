#include "GAE.h"
#include "QuadraticCost.h"
#include "RuleFactory.h"

using namespace Coeus;

GAE::GAE(NeuralNetwork* p_network, const float p_gamma, const float p_lambda) :
	_network(p_network),
	_update_rule(nullptr),
	_gamma(p_gamma),
	_lambda(p_lambda)
{
	_network_gradient = new NetworkGradient(p_network);
}

GAE::GAE(NeuralNetwork* p_network, const GRADIENT_RULE p_rule, const float p_alpha, const float p_gamma, const float p_lambda) :
	_network(p_network),
	_gamma(p_gamma),
	_lambda(p_lambda)
{
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_rule, p_network, p_alpha);
}


GAE::~GAE()
{
	delete _network_gradient;
	delete _update_rule;
}

Tensor& GAE::get_advantages(vector<DQItem> &p_sample)
{	
	const float gl = _gamma * _lambda;

	_input_buffer_s0 = Tensor({static_cast<int>(p_sample.size() + 1), _network->get_input_dim() }, Tensor::ZERO);
	_returns = Tensor({static_cast<int>(p_sample.size())}, Tensor::ZERO);
	_advantages = Tensor({static_cast<int>(p_sample.size())}, Tensor::ZERO);

	for(auto s : p_sample)
	{
		_input_buffer_s0.insert_row(&s.s0);
	}
	_input_buffer_s0.insert_row(&p_sample[p_sample.size() - 1].s1);

	_network->activate(&_input_buffer_s0);
	const Tensor Vs0 = *_network->get_output();

	float gae = 0;
	float gae_mean = 0;
	float gae_std = 0;

	for(int i = p_sample.size() - 1; i >= 0; i--)
	{
		const float delta = p_sample[i].final ? p_sample[i].r - Vs0[i] : p_sample[i].r + _gamma * Vs0[i+1] - Vs0[i];
		gae = p_sample[i].final ? delta : delta + gl * gae;
		gae_mean += gae;
		_advantages[i] = gae;
		_returns[i] = gae + Vs0[i];
	}
	gae_mean /= p_sample.size();

	for (int i = 0; i < p_sample.size(); i++)
	{
		gae_std += pow(_advantages[i] - gae_mean, 2);
	}
	gae_std = sqrt(gae_std / p_sample.size());

	_advantages = (_advantages - gae_mean) / (gae_std + 1e-10f);
	
	return _advantages;
}

void GAE::train(vector<DQItem> &p_sample)
{
	activate(p_sample);

	QuadraticCost mse;
	Tensor loss = mse.cost_deriv(_network->get_output(), &_returns);

	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient());
	_network->update(_update_rule->get_update());
}

Gradient& GAE::get_gradient(Tensor* p_state0, const float p_return) const
{
	_network->activate(p_state0);
	const float Vs0 = _network->get_output()->at(0);

	Tensor loss({ 1 }, Tensor::VALUE, Vs0 - p_return);

	_network_gradient->calc_gradient(&loss);

	return _network_gradient->get_gradient();
}

void GAE::activate(vector<DQItem>& p_sample) const
{
	Tensor input = Tensor({static_cast<int>(p_sample.size()), _network->get_input_dim() }, Tensor::ZERO);

	for(auto s : p_sample)
	{
		input.insert_row(&s.s0);		
	}

	_network->activate(&input);
}
