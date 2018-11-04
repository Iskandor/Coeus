#include "ADAMRule.h"

using namespace  Coeus;

ADAMRule::ADAMRule(NetworkGradient* p_network_gradient, double p_alpha, double p_beta1, double p_beta2, double p_epsilon) : IUpdateRule(p_network_gradient, p_alpha), 
	_beta1(p_beta1), _pow_beta1(p_beta1), _beta2(p_beta2), _pow_beta2(p_beta2), _epsilon(p_epsilon)
{
	_m = p_network_gradient->get_empty_params();
	_m_mean = p_network_gradient->get_empty_params();
	_v = p_network_gradient->get_empty_params();
	_v_mean = p_network_gradient->get_empty_params();
}

ADAMRule::~ADAMRule()
= default;

IUpdateRule* ADAMRule::clone(NetworkGradient* p_network_gradient)
{
	return new ADAMRule(p_network_gradient, _alpha, _beta1, _beta2, _epsilon);
}

void ADAMRule::update_momentum(const string& p_id, Tensor& p_gradient) {
	/*
	Tensor* m = &_m[p_id];
	Tensor* v = &_v[p_id];
	Tensor* m_mean = &_m_mean[p_id];
	Tensor* v_mean = &_v_mean[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
		(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(p_gradient[i], 2);
		(*m_mean)[i] = (*m)[i] / (1 - _beta1);
		(*v_mean)[i] = (*v)[i] / (1 - _beta2);
	}
	*/


	_m[p_id] = _beta1 * _m[p_id] + (1 - _beta1) * p_gradient;
	_v[p_id] = _beta2 * _v[p_id] + (1 - _beta2) * p_gradient.pow(2);

	if (_pow_beta1 > 1e-3) {
		_m_mean[p_id] = _m[p_id] / (1 - _pow_beta1);
		_pow_beta1 *= _beta1;
	}
	else
	{
		_m_mean[p_id] = _m[p_id];
	}

	if (_pow_beta2 > 1e-3) {
		_v_mean[p_id] = _v[p_id] / (1 - _pow_beta2);
		_pow_beta2 *= _beta2;
	}
	else
	{
		_v_mean[p_id] = _v[p_id];
	}
}

void ADAMRule::calc_update(map<string, Tensor>* p_gradient) {
	IUpdateRule::calc_update(p_gradient);

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];


		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (*m_mean)[i];
		}

		//_update[it->first] = -_alpha * _m_mean[it->first] / (_v_mean[it->first].sqrt() + _epsilon);
	}
}