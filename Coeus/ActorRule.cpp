#include "ActorRule.h"
#include "TensorOperator.h"

using namespace Coeus;

ActorRule::ActorRule(NetworkGradient* p_network_gradient, IUpdateRule* p_rule, const float p_alpha) : IUpdateRule(p_network_gradient, p_alpha), 
	_rule(p_rule)
{
}


ActorRule::~ActorRule()
{
	delete _rule;
}

void ActorRule::calc_update(map<string, Tensor>* p_gradient, const float p_delta, Tensor* p_policy, float p_alpha)
{
	IUpdateRule::calc_update(p_gradient, p_alpha);

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
			TensorOperator::instance().vc_prod_add(gsb[i][it->first].arr(), p_policy->at(i), g.arr(), g.size());
		}

		TensorOperator::instance().vv_sub(gsa[it->first].arr(), g.arr(), _update[it->first].arr(), g.size());
		TensorOperator::instance().vc_prod(_update[it->first].arr(), -p_delta, _update[it->first].arr(), _update[it->first].size());
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
