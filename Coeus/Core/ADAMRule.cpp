#include "ADAMRule.h"
#include <emmintrin.h>
#include <immintrin.h>

using namespace  Coeus;

ADAMRule::ADAMRule(ParamModel* p_model, const float p_alpha, const float p_beta1, const float p_beta2, const float p_epsilon) :
	IUpdateRule(p_model, p_alpha), _t(0),
	_beta1(p_beta1), _denb1(0), _beta2(p_beta2), _denb2(0), _epsilon(p_epsilon)
{
	_m = p_model->get_empty_params();
	_m_mean = p_model->get_empty_params();
	_v = p_model->get_empty_params();
	_v_mean = p_model->get_empty_params();
}

ADAMRule::~ADAMRule()
= default;

IUpdateRule* ADAMRule::clone(ParamModel* p_model)
{
	return new ADAMRule(p_model, _alpha, _beta1, _beta2, _epsilon);
}

void ADAMRule::calc_update(Gradient& p_gradient, float p_alpha) {
	
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

	float ib1 = 1 - _beta1;
	float ib2 = 1 - _beta2;
	float alpha = -_alpha;
	
	__m128 vec_beta1 = _mm_broadcast_ss(&_beta1);	
	__m128 vec_ibeta1 = _mm_broadcast_ss(&ib1);
	__m128 vec_beta2 = _mm_broadcast_ss(&_beta2);
	__m128 vec_ibeta2 = _mm_broadcast_ss(&ib2);
	__m128 vec_denb1 = _mm_broadcast_ss(&_denb1);
	__m128 vec_denb2 = _mm_broadcast_ss(&_denb2);
	__m128 vec_alpha = _mm_broadcast_ss(&alpha);
	__m128 vec_epsilon = _mm_broadcast_ss(&_epsilon);

	for (auto it = p_gradient.begin(); it != p_gradient.end(); ++it) {
		
		float *g = (p_gradient)[it->first].arr();
		float *m = (_m[it->first].arr());
		float *v = (_v[it->first].arr());
		float *m_mean = (_m_mean[it->first].arr());
		float *v_mean = (_v_mean[it->first].arr());
		float *update = (_update[it->first].arr());
		
		int size = it->second.size() / 4 * 4;
		//int size = 0;
		
		for (int i = 0; i < size; i+=4) {
			__m128 gx = _mm_load_ps(g);
			__m128 mx = _mm_load_ps(m);
			__m128 vx = _mm_load_ps(v);
			__m128 mmx = _mm_load_ps(m_mean);
			__m128 vmx = _mm_load_ps(v_mean);
			__m128 ux = _mm_load_ps(update);

			mx = _mm_add_ps(_mm_mul_ps(vec_beta1, mx), _mm_mul_ps(vec_ibeta1, gx));
			vx = _mm_add_ps(_mm_mul_ps(vec_beta2,  vx), _mm_mul_ps(vec_ibeta2, _mm_mul_ps(gx, gx)));

			if (_t < 1e4)
			{
				mmx = _mm_div_ps(mx, vec_denb1);
				vmx = _mm_div_ps(vx, vec_denb2);
				
				ux = _mm_div_ps(_mm_mul_ps(vec_alpha, mmx), _mm_add_ps(_mm_sqrt_ps(vmx), vec_epsilon));
			}
			else
			{
				ux = _mm_div_ps(_mm_mul_ps(vec_alpha, mx), _mm_add_ps(_mm_sqrt_ps(vx), vec_epsilon));
			}

			_mm_store_ps(m, mx);
			_mm_store_ps(v, vx);
			_mm_store_ps(m_mean, mmx);
			_mm_store_ps(v_mean, vmx);
			_mm_store_ps(update, ux);
			
			g += 4;
			m += 4;
			v += 4;
			m_mean += 4;
			v_mean += 4;
			update += 4;
		}

		for (int i = size; i < it->second.size(); i++) {
			*m = _beta1 * *m + ib1 * *g;
			*v = _beta2 * *v + ib2 * (*g * *g);

			if (_t < 1e4)
			{
				*m_mean = *m++ / _denb1;
				*v_mean = *v++ / _denb2;

				*update++ = -_alpha * *m_mean++ / (sqrt(*v_mean++) + _epsilon);
			}
			else
			{
				*update++ = -_alpha * *m++ / (sqrt(*v++) + _epsilon);
			}

			g++;
		}

		
	}
}