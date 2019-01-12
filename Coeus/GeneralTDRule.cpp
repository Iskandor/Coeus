#include "GeneralTDRule.h"

using namespace Coeus;

GeneralTDRule::GeneralTDRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, const double p_alpha, const double p_gamma, const double p_lambda) : IUpdateRule(p_network_gradient, p_alpha),
	_gamma(p_gamma),
	_lambda(p_lambda),
	_rule(p_rule)
{
	if (p_lambda > 0)
	{
		_e_traces = p_network_gradient->get_empty_params();
	}
}


GeneralTDRule::~GeneralTDRule()
{
	delete _rule;
}


void GeneralTDRule::calc_update(map<string, Tensor>* p_gradient, const double p_delta)
{
	IUpdateRule::calc_update(p_gradient);

	_rule->calc_update(p_gradient);

	if (_lambda > 0)
	{
		update_traces(_rule->get_update());
	}

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

IUpdateRule* GeneralTDRule::clone(NetworkGradient* p_network_gradient)
{
	return new GeneralTDRule(p_network_gradient, _rule->clone(p_network_gradient), _alpha, _gamma, _lambda);
}

void GeneralTDRule::reset_traces()
{
	for (auto it = _e_traces.begin(); it != _e_traces.end(); ++it) {
		it->second.fill(0);
	}
}

void GeneralTDRule::update_traces(map<string, Tensor>* p_gradient)
{
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		_e_traces[it->first] = _lambda * _gamma * _e_traces[it->first] + it->second;
	}
}
