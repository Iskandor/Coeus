#include "QLearningRule.h"

using namespace Coeus;

QLearningRule::QLearningRule(NetworkGradient* p_network_gradient, const float p_alpha, const float p_gamma, const float p_lambda) : IUpdateRule(p_network_gradient, p_alpha),
	_gamma(p_gamma),
	_lambda(p_lambda)
{
	if (p_lambda > 0)
	{
		_e_traces = p_network_gradient->get_empty_params();
	}

	_rule = new ADAMRule(p_network_gradient, p_alpha, 0.9, 0.999, 1e-8);
}


QLearningRule::~QLearningRule()
= default;

void QLearningRule::calc_update(map<string, Tensor>* p_gradient, const float p_delta)
{
	IUpdateRule::calc_update(p_gradient);

	_rule->calc_update(p_gradient);

	if (_lambda > 0)
	{
		update_traces(_rule->get_update());
	}

	//cout << p_delta << endl;

	

	/*
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		if (_lambda > 0)
		{
			_update[it->first] = _alpha * p_delta * _e_traces[it->first];
		}
		else
		{
			//cout << it->second << endl;
			_update[it->first] = _alpha * p_delta * it->second;
		}
		
	}
	*/

	for (auto it = _rule->get_update()->begin(); it != _rule->get_update()->end(); ++it) {
		if (_lambda > 0)
		{
			_update[it->first] = -p_delta * _e_traces[it->first];
		}
		else
		{
			_update[it->first] = -p_delta * it->second;
		}
	}
}

IUpdateRule* QLearningRule::clone(NetworkGradient* p_network_gradient)
{
	return new QLearningRule(p_network_gradient, _alpha, _gamma, _lambda);
}

void QLearningRule::reset()
{
	_rule->reset();
}

void QLearningRule::reset_traces()
{
	for (auto it = _e_traces.begin(); it != _e_traces.end(); ++it) {
		it->second.fill(0);
	}
}

void QLearningRule::update_traces(map<string, Tensor>* p_gradient)
{
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		_e_traces[it->first] = _lambda * _gamma * _e_traces[it->first] + it->second;
	}
}
