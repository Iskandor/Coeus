#include "ADAMRule.h"

using namespace  Coeus;

ADAMRule::ADAMRule(NetworkGradient* p_network_gradient, float p_alpha, float p_beta1, float p_beta2, float p_epsilon) :
	IUpdateRule(p_network_gradient, p_alpha), _t(0),
	_beta1(p_beta1), _denb1(0), _beta2(p_beta2), _denb2(0), _epsilon(p_epsilon)
{
	_m = p_network_gradient->get_network()->get_empty_params();
	_m_mean = p_network_gradient->get_network()->get_empty_params();
	_v = p_network_gradient->get_network()->get_empty_params();
	_v_mean = p_network_gradient->get_network()->get_empty_params();
}

ADAMRule::~ADAMRule()
= default;

IUpdateRule* ADAMRule::clone(NetworkGradient* p_network_gradient)
{
	return new ADAMRule(p_network_gradient, _alpha, _beta1, _beta2, _epsilon);
}

void ADAMRule::calc_update(map<string, Tensor>* p_gradient, float p_alpha) {
	
	if (p_alpha > 0)
	{
		_alpha = p_alpha;
	}

	if (_t < 1e4)
	{
		_t++;
		_denb1 = 1 - pow(_beta1, _t);
		_denb2 = 1 - pow(_beta2, _t);
	}
	
	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {
		Tensor* g = &(*p_gradient)[it->first];
		Tensor* m = &_m[it->first];
		Tensor* v = &_v[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];
		Tensor* update = &_update[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * (*g)[i];
			(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow((*g)[i], 2);

			if (_t < 1e4)
			{
				(*m_mean)[i] = (*m)[i] / _denb1;
				(*v_mean)[i] = (*v)[i] / _denb2;
				(*update)[i] = -_alpha * (*m_mean)[i] / (sqrt((*v_mean)[i]) + _epsilon);
			}
			else
			{
				(*update)[i] = -_alpha * (*m)[i] / (sqrt((*v)[i]) + _epsilon);
			}

		}
	}
}