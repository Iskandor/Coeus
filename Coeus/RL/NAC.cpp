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
	_gradient_actor = new NaturalGradient(_network_actor);
	_rule_actor = RuleFactory::create_rule(BACKPROP_RULE, p_network_actor, p_alpha_actor);
	_actor_loss = Tensor::Zero({_network_actor->get_output_dim()});
	_gradient_estimate = _network_actor->get_empty_params();
	_update = _network_actor->get_empty_params();
}

NAC::~NAC()
{
	delete _critic;
	delete _gradient_actor;
	delete _rule_actor;
}

void NAC::add_sample(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, float p_r, const bool p_final)
{
	_sample_buffer.emplace_back(p_s0, p_a, p_s1, p_r, p_final);
}

void NAC::train()
{
	NaturalGradient* gradient = static_cast<NaturalGradient*>(_gradient_actor);
	//map<string, Tensor> gradient_estimate = _network_actor->get_empty_params();
	
	_critic->set_sample(_sample_buffer);
	vector<float> advantage = _critic->get_advantages();

	for(int i = 0; i < _sample_buffer.size(); i++)
	{
		_network_actor->activate(&_sample_buffer[i].s0);
		//cout << _sample_buffer[i].s0 << endl;
		cout << *_network_actor->get_output() << endl;
		
		_actor_loss.fill(0);
		_actor_loss[_sample_buffer[i].a.max_value_index()] = -advantage[i] / _network_actor->get_output()->at(_sample_buffer[i].a.max_value_index());

		gradient->calc_gradient(&_actor_loss);

		for (const auto& g : gradient->get_regular_gradient()) {
			_gradient_estimate[g.first] += (1.f / _sample_buffer.size()) *  g.second;
		}
	}
	
	for(const auto& g : gradient->get_regular_gradient())
	{
		//Tensor temp = g.second;
		const int rows = g.second.shape(0);
		const int cols = g.second.shape(1);

		/*
		TensorOperator::instance().vv_sub(g.second.arr(), _gradient_estimate[g.first].arr(), temp.arr(), rows * cols);
		TensorOperator::instance().vv_add(_gradient_estimate[g.first].arr(), 1, temp.arr(), _epsilon, _gradient_estimate[g.first].arr(), rows * cols);
		*/
		float c = sqrt(2 * _delta / _gradient_estimate[g.first].vec().dot(gradient->get_hessian_inv()[g.first] * _gradient_estimate[g.first].vec()));
		_update[g.first] = c * gradient->get_hessian_inv()[g.first] * _gradient_estimate[g.first].vec();
		_update[g.first].reshape({rows, cols});

		/*
		float c = sqrt(2 * _delta / g.second.vec().dot(gradient->get_hessian_inv()[g.first] * g.second.vec()));
		_update[g.first] = c * gradient->get_hessian_inv()[g.first] * g.second.vec();
		_update[g.first].reshape({ rows, cols });
		*/
	}
	
	//_critic->train();
	_network_actor->update(&_update);	
	_sample_buffer.clear();
}
