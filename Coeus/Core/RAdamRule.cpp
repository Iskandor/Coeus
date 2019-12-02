#include "RAdamRule.h"

using namespace Coeus;

RADAMRule::RADAMRule(ParamModel* p_model, float p_alpha, float p_beta1, float p_beta2): IUpdateRule(p_model, p_alpha),
	_t(0), _beta1(p_beta1), _denb1(0), _beta2(p_beta2), _denb2(0), _rho(0), _r(0)
{
	_m = p_model->get_empty_params();
	_m_mean = p_model->get_empty_params();
	_v = p_model->get_empty_params();
	_v_mean = p_model->get_empty_params();

	_rho_inf = 2 / (1 - _beta2) - 1;
}

RADAMRule::~RADAMRule()
= default;

IUpdateRule* RADAMRule::clone(ParamModel* p_model)
{
	return new RADAMRule(p_model, _alpha, _beta1, _beta2);
}

void RADAMRule::calc_update(Gradient& p_gradient, float p_alpha)
{
	if (p_alpha > 0)
	{
		_alpha = p_alpha;
	}

	_t++;
	const float powb1 = pow(_beta1, _t);
	const float powb2 = pow(_beta2, _t);
	_denb1 = 1 - powb1;
	_denb2 = 1 - powb2;

	_rho = _rho_inf - 2.f * _t * powb2 / _denb2;
	
	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {
		Tensor* g = &(p_gradient)[it->first];
		Tensor* m = &_m[it->first];
		Tensor* v = &_v[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];
		Tensor* update = &_update[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * (*g)[i];
			(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow((*g)[i], 2);

			(*m_mean)[i] = (*m)[i] / _denb1;
			
			if (_rho > 4)
			{
				(*v_mean)[i] = sqrt((*v)[i] / _denb2);
				_r = sqrt(((_rho - 4) * (_rho - 2) * _rho_inf) / ((_rho_inf - 4) * (_rho_inf - 2) * _rho));
				(*update)[i] = -_alpha * _r * (*m_mean)[i] / ((*v_mean)[i] + 1e-8f);
			}
			else
			{
				(*update)[i] = -_alpha * (*m_mean)[i];
			}
		}
	}
}
