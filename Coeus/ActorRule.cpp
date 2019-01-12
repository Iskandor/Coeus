#include "ActorRule.h"

using namespace Coeus;

ActorRule::ActorRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, const double p_alpha) : IUpdateRule(p_network_gradient, p_alpha), 
	_rule(p_rule)
{
}


ActorRule::~ActorRule()
{
	delete _rule;
}

void ActorRule::calc_update(map<string, Tensor>* p_gradient, const double p_delta, Tensor* p_policy)
{
	IUpdateRule::calc_update(p_gradient);

	_rule->calc_update(p_gradient);
	map<string, Tensor> gsa = *_rule->get_update();
	const auto gsb = new map<string, Tensor>[p_policy->size()];

	Tensor mask = Tensor::Zero({ p_policy->size() });
	
	for (int i = 0; i < p_policy->size(); i++)
	{
		mask.fill(0);
		mask[i] = 1;
		_network_gradient->calc_gradient(&mask);
		_rule->calc_update(_network_gradient->get_gradient());
		gsb[i] = *_rule->get_update();
	}

	for (auto it = gsa.begin(); it != gsa.end(); ++it) {

		Tensor g(it->second.rank(), it->second.shape(), Tensor::ZERO);

		for(int i = 0; i < p_policy->size(); i++)
		{

			g += gsb[i][it->first] * p_policy->at(i);
		}

		_update[it->first] = -p_delta * (gsa[it->first] - g);
	}

	delete[] gsb;
}

IUpdateRule* ActorRule::clone(NetworkGradient* p_network_gradient)
{
	return new ActorRule(p_network_gradient, _rule->clone(p_network_gradient), _alpha);
}

void ActorRule::reset()
{
}
