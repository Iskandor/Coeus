#include "AdadeltaRule.h"

using namespace Coeus;

AdadeltaRule::AdadeltaRule(ParamModel* p_model, const float p_alpha, const float p_decay, const float p_epsilon): IUpdateRule(p_model, p_alpha),
	_decay(p_decay), _epsilon(p_epsilon)
{
	_cache = p_model->get_empty_params();
	_cache_delta = p_model->get_empty_params();
}

AdadeltaRule::~AdadeltaRule()
= default;

IUpdateRule* AdadeltaRule::clone(ParamModel* p_model)
{
	return new AdadeltaRule(p_model, _alpha, _decay, _epsilon);
}

void AdadeltaRule::update_cache(const string& p_id, Tensor& p_gradient) {
	Tensor* cache = &_cache[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*cache)[i] = _decay * (*cache)[i] + (1 - _decay) * pow(p_gradient[i], 2);
	}
}

void AdadeltaRule::update_cache_delta(const string& p_id, Tensor& p_gradient) {
	Tensor* cache = &_cache[p_id];
	Tensor* cache_delta = &_cache_delta[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*cache_delta)[i] = _decay * (*cache_delta)[i] + (1 - _decay) * pow(_alpha / sqrt((*cache)[i] + _epsilon) * p_gradient[i], 2);
	}
}

void AdadeltaRule::calc_update(map<string, Tensor>& p_gradient, const float p_alpha) {
	IUpdateRule::calc_update(p_gradient, p_alpha);
	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {
		update_cache(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* cache = &_cache[it->first];
		Tensor* cache_delta = &_cache_delta[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -sqrt((*cache_delta)[i] + _epsilon) / sqrt((*cache)[i] + _epsilon) * it->second[i];
		}

		update_cache_delta(it->first, it->second);
	}
}