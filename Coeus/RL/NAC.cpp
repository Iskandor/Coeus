#include "NAC.h"
#include "RuleFactory.h"
#include "NaturalGradient.h"
#include "TensorOperator.h"
#include <omp.h>

using namespace Coeus;

NAC::NAC(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork* p_network_actor, float p_delta):
	_delta(p_delta)
{
	_network_actor = p_network_actor;
	_network_critic = p_network_critic;
	_actor_natural_gradient = new NaturalGradient(p_network_actor);
	_rule_critic = RuleFactory::create_rule(p_rule_critic, p_network_critic, p_alpha_critic);
	_rule_actor = RuleFactory::create_rule(BACKPROP_RULE, p_network_actor, 1.f);

	_critic = new GAE(p_network_critic, p_gamma, p_lambda);
	_actor = new PolicyGradient(_network_actor);

	_actor_update = p_network_actor->get_empty_params();
	_critic_gradient.init(_network_critic);
}

NAC::~NAC()
{
	delete _critic;
	delete _actor;
	delete _actor_natural_gradient;
	delete _rule_critic;
	delete _rule_actor;
}

void NAC::add_sample(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, float p_r, const bool p_final)
{
	_sample_buffer.emplace_back(p_s0, p_a, p_s1, p_r, p_final);
}

void NAC::train()
{
	_critic->set_sample(_sample_buffer);
	vector<float> advantage = _critic->get_advantages();
	
	for (const auto& p : _critic_gradient)
	{
		p.second.fill(0);
	}
	
	for(const auto& p : _actor_update)
	{
		p.second.fill(0);
	}

	for(size_t i = 0; i < _sample_buffer.size(); i++)
	{
		_critic_gradient += _critic->get_gradient(&_sample_buffer[i].s0, advantage[i]);

		Gradient& g = _actor->get_gradient(&_sample_buffer[i].s0, _sample_buffer[i].a.max_value_index(), advantage[i]);		

		_actor_natural_gradient->calc_hessian(g);
		Gradient& ag = _actor_natural_gradient->get_gradient();

		map<string, Tensor>& H = _actor_natural_gradient->get_hessian_inv();

		for (auto& p : _actor_update)
		{
			float c = (g[p.first].vec().T() * H[p.first]).dot(g[p.first].vec());
			c = c == 0 ? sqrt(2 * _delta) : sqrt(2 * _delta / c);
			p.second += c * ag[p.first];
			//cout << p.second.max_value() << ",";
		}
	}
	//cout << endl;
	
	
	_rule_critic->calc_update(_critic_gradient);
	_network_critic->update(_rule_critic->get_update());

	_network_actor->update(&_actor_update);
		
	_sample_buffer.clear();
}
