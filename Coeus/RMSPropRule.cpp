#include "RMSPropRule.h"

using namespace Coeus;

RMSPropRule::RMSPropRule(ParamModel* p_model, const float p_alpha, const float p_decay, const float p_epsilon):
	IUpdateRule(p_model, p_alpha), _decay(p_decay), _epsilon(p_epsilon)
{
	_cache = p_model->get_empty_params();
}

RMSPropRule::~RMSPropRule()
= default;

void RMSPropRule::calc_update(Gradient& p_gradient, const float p_alpha) {
	IUpdateRule::calc_update(p_gradient, p_alpha);
	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {

		Tensor* cache = &_cache[it->first];
		Tensor* update = &_update[it->first];

		for(int i = 0; i < it->second.size(); i++) {
			(*cache)[i] = _decay * (*cache)[i] + (1 - _decay) * pow(it->second[i], 2);
			(*update)[i] = -_alpha / sqrt((*cache)[i] + _epsilon) * it->second[i];
		}
	}
}

IUpdateRule* RMSPropRule::clone(ParamModel* p_model)
{
	return new RMSPropRule(p_model, _alpha, _decay, _epsilon);
}