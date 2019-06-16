#include "GeneralTDRule.h"
#include "TensorOperator.h"

using namespace Coeus;

GeneralTDRule::GeneralTDRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, const float p_alpha, const float p_gamma, const float p_lambda) : IUpdateRule(p_network_gradient, p_alpha),
	_gamma(p_gamma),
	_lambda(p_lambda),
	_rule(p_rule)
{
	if (p_lambda > 0)
	{
		_e_traces = p_network_gradient->get_network()->get_empty_params();
	}
}


GeneralTDRule::~GeneralTDRule()
{
	delete _rule;
}

void GeneralTDRule::calc_update(map<string, Tensor>* p_gradient, const float p_delta, const float p_alpha)
{
	_delta = p_delta;
	calc_update(p_gradient, p_alpha);
}


void GeneralTDRule::calc_update(map<string, Tensor>* p_gradient, const float p_alpha)
{
	IUpdateRule::calc_update(p_gradient, p_alpha);

	_rule->calc_update(p_gradient);
	_update = *_rule->get_update();

	if (_lambda > 0)
	{
		update_traces(p_gradient);
	}

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		if (_lambda > 0)
		{
			TensorOperator::instance().vc_prod(_e_traces[it->first].arr(), -_delta, _update[it->first].arr(), _update[it->first].size());
		}
		else
		{
			TensorOperator::instance().vc_prod(_update[it->first].arr(), -_delta, _update[it->first].arr(), _update[it->first].size());
		}
	}
}

IUpdateRule* GeneralTDRule::clone(NetworkGradient* p_network_gradient)
{
	return new GeneralTDRule(p_network_gradient, _rule->clone(p_network_gradient), _alpha, _gamma, _lambda);
}

void GeneralTDRule::reset()
{
}

void GeneralTDRule::reset_traces()
{
	for (auto it = _e_traces.begin(); it != _e_traces.end(); ++it) {
		it->second.fill(0);
	}
}

void GeneralTDRule::update_traces(map<string, Tensor>* p_gradient)
{
	for (auto& it : *p_gradient)
	{
		const int size = _e_traces[it.first].size();
		TensorOperator::instance().vc_prod(_e_traces[it.first].arr(), _lambda * _gamma, _e_traces[it.first].arr(), size);
		TensorOperator::instance().vv_add(_e_traces[it.first].arr(), it.second.arr(), _e_traces[it.first].arr(), size);
	}
}
