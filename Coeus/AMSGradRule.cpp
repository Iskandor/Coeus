#include "AMSGradRule.h"
#include <algorithm>

using namespace Coeus;

AMSGradRule::AMSGradRule(ParamModel* p_model, const float p_alpha, const float p_beta1, const float p_beta2,
                         const float p_epsilon): IUpdateRule(p_model, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
	_m = p_model->get_empty_params();
	_v = p_model->get_empty_params();
	_v_mean = p_model->get_empty_params();
}

AMSGradRule::~AMSGradRule()
= default;

IUpdateRule* AMSGradRule::clone(ParamModel* p_model)
{
	return new AMSGradRule(p_model, _alpha, _beta1, _beta2, _epsilon);
}

void AMSGradRule::update_momentum(const string& p_id, Tensor& p_gradient) {

	Tensor* m = &_m[p_id];
	Tensor* v = &_v[p_id];
	Tensor* v_mean = &_v_mean[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
		(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(p_gradient[i], 2);
		(*v_mean)[i] = max((*v_mean)[i], (*v)[i]);
	}

}

void AMSGradRule::calc_update(map<string, Tensor>& p_gradient, const float p_alpha) {
	IUpdateRule::calc_update(p_gradient, p_alpha);

	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {
		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m = &_m[it->first];
		Tensor* v_mean = &_v_mean[it->first];


		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (*m)[i];
		}
	}
}