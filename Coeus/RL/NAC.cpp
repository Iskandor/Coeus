#include "NAC.h"
#include "RuleFactory.h"
#include "NaturalGradient.h"
#include "TensorOperator.h"
#include <omp.h>

using namespace Coeus;

NAC::NAC(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork* p_network_actor, float p_alpha_actor, float p_delta):
	_delta(p_delta),
	_epsilon(1e-2f)
{
	_critic = new GAE(p_network_critic, p_gamma, p_lambda);

	_network_actor = p_network_actor;
	_network_critic = p_network_critic;
	_gradient_actor = new PolicyGradient(_network_actor);
	_rule_actor = RuleFactory::create_rule(BACKPROP_RULE, p_network_actor, p_alpha_actor);
	_rule_critic = RuleFactory::create_rule(p_rule_critic, p_network_critic, p_alpha_critic);
	_update = _network_actor->get_empty_params();

	for (auto& param : p_network_actor->get_empty_params())
	{
		const int row = param.second.shape(0) * param.second.shape(1);
		_fim[param.first] = Tensor({ row, row }, Tensor::ONES);
		_inv_fim[param.first] = Tensor({ row, row }, Tensor::ZERO);
	}
}

NAC::~NAC()
{
	delete _critic;
	delete _gradient_actor;
	delete _rule_actor;
	delete _rule_critic;
}

void NAC::add_sample(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, float p_r, const bool p_final)
{
	_sample_buffer.emplace_back(p_s0, p_a, p_s1, p_r, p_final);
}

void NAC::train()
{
	_critic->set_sample(_sample_buffer);
	vector<float> advantage = _critic->get_advantages();

	for(int i = 0; i < _sample_buffer.size(); i++)
	{
		_network_actor->activate(&_sample_buffer[i].s0);
		cout << *_network_actor->get_output() << endl;
		
		Gradient actor_grad = _gradient_actor->get_gradient(&_sample_buffer[i].s0, _sample_buffer[i].a.max_value_index(), advantage[i]);

		for (auto& g : actor_grad)
		{
			_fim[g.first] += g.second.vec() * g.second.vec().T();
			//_fim[g.first] /= i+1;
			_inv_fim[g.first] = _fim[g.first].inv();
			//Tensor temp = g.second;

			/*
			TensorOperator::instance().vv_sub(g.second.arr(), _gradient_estimate[g.first].arr(), temp.arr(), rows * cols);
			TensorOperator::instance().vv_add(_gradient_estimate[g.first].arr(), 1, temp.arr(), _epsilon, _gradient_estimate[g.first].arr(), rows * cols);

			float c = sqrt(2 * _delta / g.second.vec().dot(gradient->get_hessian_inv()[g.first] * g.second.vec()));
			cout << c << endl;
			_update[g.first] += c * gradient->get_hessian_inv()[g.first] * g.second.vec();

			float c = sqrt(2 * _delta / g.second.vec().dot(gradient->get_hessian_inv()[g.first] * g.second.vec()));
			_update[g.first] = c * gradient->get_hessian_inv()[g.first] * g.second.vec();
			_update[g.first].reshape({ rows, cols });
			*/
			const float c = advantage[i] == 0 ? 1 : sqrt(2 * _delta / g.second.vec().dot(_inv_fim[g.first] * g.second.vec()));
			//cout << g.first << " " << c << endl;
			_update[g.first] += c * _inv_fim[g.first] * g.second.vec();
		}

		Gradient critic_grad = _critic->get_gradient(&_sample_buffer[i].s0, advantage[i]);
		_rule_critic->calc_update(critic_grad);
		_network_critic->update(_rule_critic->get_update());
	}

	for (auto& g : _network_actor->get_empty_params())
	{
		const int rows = g.second.shape(0);
		const int cols = g.second.shape(1);

		_update[g.first].reshape({ rows, cols });
	}
	
	_network_actor->update(&_update);	
	_sample_buffer.clear();
}
