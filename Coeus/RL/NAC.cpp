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
	_actor_gradient.init(_network_actor);
	_fisher_matrix = Tensor({ p_network_actor->get_params_size(), p_network_actor->get_params_size() }, Tensor::ZERO);
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
	//_critic_gradient.fill(0);
	_actor_gradient.fill(0);
	_fisher_matrix.fill(0);
	
	_critic->set_sample(_sample_buffer);
	vector<float> advantage = _critic->get_advantages();
	
	for(size_t i = 0; i < _sample_buffer.size(); i++)
	{
		//Gradient& actor_gradient = _actor->get_gradient(&_sample_buffer[i].s0, _sample_buffer[i].a.max_value_index(), advantage[i]);
		//NaturalGradient::calc_hessian(actor_gradient);
		
		_rule_critic->calc_update(_critic->get_gradient(&_sample_buffer[i].s0, advantage[i]));
		_network_critic->update(_rule_critic->get_update());

		//_actor_gradient += actor_gradient;		
		//_fisher_matrix += actor_gradient.hessian();
	}
	/*
	_actor_gradient /= _sample_buffer.size();
	_fisher_matrix /= _sample_buffer.size();
	_actor_gradient.flatten();
	
	const Tensor Finv = _fisher_matrix.pinv();
	//cout << Finv.has_NaN_Inf() << endl;
	Tensor Finvg = Finv * _actor_gradient.gradient();

	const float lr_den = _actor_gradient.gradient().T().dot(Finvg);
	const float lr = lr_den == 0 ? 0 : sqrt(_delta / lr_den);
	cout << lr << endl;

	_actor_gradient.fill(Finvg);
	
	//cout << endl;
	
	for (auto& actor_update_param : _actor_update)
	{
		actor_update_param.second = lr * _actor_gradient[actor_update_param.first];
	}
	//_network_actor->update(&_actor_update);
	*/
		
	_sample_buffer.clear();
}
