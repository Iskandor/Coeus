#include "adam.h"

#include <iostream>

#include "CLAB.h"


adam::adam(neural_network* p_model, const float p_alpha, float p_weight_decay, const float p_beta1, const float p_beta2, const float p_epsilon) : optimizer(p_model, p_alpha, p_weight_decay),
	_t(0),
	_beta1(p_beta1),
	_beta2(p_beta2),
	_epsilon(p_epsilon)
{
	_m = p_model->zero_model();
	_v = p_model->zero_model();
}

adam::~adam()
= default;

void adam::update()
{
	optimizer::update();

	float denb1 = 1;
	float denb2 = 1;

	_t = 1e5;
	
	if (_t < 1e4)
	{
		_t++;
		denb1 -= pow(_beta1, _t);
		denb2 -= pow(_beta2, _t);
	}

	float ib1 = 1 - _beta1;
	float ib2 = 1 - _beta2;


	const __m256 alpha256 = _mm256_broadcast_ss(&_alpha);
	const __m256 ib1256 = _mm256_broadcast_ss(&ib1);
	const __m256 ib2256 = _mm256_broadcast_ss(&ib2);
	const __m256 denb1256 = _mm256_broadcast_ss(&denb1);
	const __m256 denb2256 = _mm256_broadcast_ss(&denb2);
	const __m256 beta1256 = _mm256_broadcast_ss(&_beta1);
	const __m256 beta2256 = _mm256_broadcast_ss(&_beta2);
	const __m256 epsilon256 = _mm256_broadcast_ss(&_epsilon);

	for (auto& param : *_model)
	{
		const int size = param.second->gradient().size() / segment;

		tensor test1 = tensor::zero_like(param.second->params());
		
		float* mx = _m[param.first].data();
		float* vx = _v[param.first].data();
		float* px = param.second->params().data();
		float* gx = param.second->gradient().data();

		for (int i = 0; i < size; i++) {
			const __m256 gx256 = _mm256_load_ps(gx);
			__m256 px256 = _mm256_load_ps(px);
			__m256 mx256 = _mm256_load_ps(mx);
			__m256 vx256 = _mm256_load_ps(vx);

			mx256 = _mm256_add_ps(_mm256_mul_ps(beta1256, mx256), _mm256_mul_ps(ib1256, gx256));
			vx256 = _mm256_add_ps(_mm256_mul_ps(beta2256, vx256), _mm256_mul_ps(ib2256, _mm256_mul_ps(gx256, gx256)));
			
			if (_t < 1e4)
			{
				const __m256 m_meanx256 = _mm256_div_ps(mx256, denb1256);
				__m256 v_meanx256 = _mm256_div_ps(vx256, denb2256);
				
				for (float& vxi : v_meanx256.m256_f32)
				{
					vxi = sqrt(vxi);
				}
				px256 = _mm256_sub_ps(px256, _mm256_mul_ps(alpha256, _mm256_div_ps(m_meanx256, _mm256_add_ps(v_meanx256, epsilon256))));
			}
			else
			{
				__m256 sqrtvx256 = vx256;
				for (float& vxi : sqrtvx256.m256_f32)
				{
					vxi = sqrt(vxi);
				}
				px256 = _mm256_sub_ps(px256, _mm256_mul_ps(alpha256, _mm256_div_ps(mx256, _mm256_add_ps(sqrtvx256, epsilon256))));
			}
			
			_mm256_storeu_ps(px, px256);
			_mm256_storeu_ps(mx, mx256);
			_mm256_storeu_ps(vx, vx256);
			
			px += segment;
			gx += segment;
			mx += segment;
			vx += segment;
		}
		
		for (int i = size * segment; i < param.second->gradient().size(); i++) {
			*mx = _beta1 * *mx + ib1 * *gx;			
			*vx = _beta2 * *vx + ib2 * (*gx * *gx);			

			if (_t < 1e4)
			{
				const float m_meanx = *mx++ / denb1;
				const float v_meanx = *vx++ / denb2;

				*px++ -= _alpha * m_meanx / (sqrt(v_meanx) + _epsilon);
			}
			else
			{
				*px++ -= _alpha * *mx++ / (sqrt(*vx++) + _epsilon);
			}

			gx++;
		}
	}
}
