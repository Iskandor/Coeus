#include "radam.h"
#include "CLAB.h"


radam::radam(neural_network* p_model, const float p_alpha, const float p_weight_decay, const float p_beta1, const float p_beta2, const float p_epsilon) : 
	adam(p_model, p_alpha, p_weight_decay, p_beta1, p_beta2, p_epsilon),
	_r(0),
	_rho(0)
{
	_rho_inf = 2 / (1 - p_beta2) - 1;
}

radam::~radam()
= default;

void radam::update()
{
	optimizer::update();

	float denb1 = 1;
	float denb2 = 1;
	float ib1 = 1 - _beta1;
	float ib2 = 1 - _beta2;

	if (_t < 1e4)
	{
		_t++;
		denb1 = 1 - pow(_beta1, _t);
		denb2 = 1 - pow(_beta2, _t);
		_rho = _rho_inf - 2.f * _t * pow(_beta2, _t) / denb2;
	}

	
	const __m256 beta1256 = _mm256_broadcast_ss(&_beta1);
	const __m256 beta2256 = _mm256_broadcast_ss(&_beta2);
	const __m256 epsilon256 = _mm256_broadcast_ss(&_epsilon);
	const __m256 ib1256 = _mm256_broadcast_ss(&ib1);
	const __m256 ib2256 = _mm256_broadcast_ss(&ib2);
	const __m256 denb1256 = _mm256_broadcast_ss(&denb1);
	const __m256 denb2256 = _mm256_broadcast_ss(&denb2);

	if (_rho > 4)
	{
		_r = sqrt(((_rho - 4) * (_rho - 2) * _rho_inf) / ((_rho_inf - 4) * (_rho_inf - 2) * _rho));
		const float alpha = _alpha * _r;
		const __m256 alpha256 = _mm256_broadcast_ss(&alpha);

		for (auto& param : *_model)
		{
			const int size = param.second->gradient().size() / segment;

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
					__m256 m_meanx256 = _mm256_div_ps(mx256, denb1256);
					__m256 v_meanx256 = _mm256_div_ps(vx256, denb2256);

					for (float& vxi : v_meanx256.m256_f32)
					{
						vxi = sqrt(vxi);
					}
					px256 = _mm256_add_ps(px256, _mm256_mul_ps(alpha256, _mm256_div_ps(m_meanx256, _mm256_add_ps(v_meanx256, epsilon256))));
				}
				else
				{
					__m256 sqrtvx256 = vx256;
					for (float& vxi : sqrtvx256.m256_f32)
					{
						vxi = sqrt(vxi);
					}
					px256 = _mm256_add_ps(px256, _mm256_mul_ps(alpha256, _mm256_div_ps(mx256, _mm256_add_ps(sqrtvx256, epsilon256))));
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

					*px++ += _alpha * m_meanx / (sqrt(v_meanx) + _epsilon);
				}
				else
				{
					*px++ += _alpha * *mx++ / (sqrt(*vx++) + _epsilon);
				}

				gx++;
			}
		}
	}
	else
	{
		const __m256 alpha256 = _mm256_broadcast_ss(&_alpha);
		for (auto& param : *_model)
		{
			const int size = param.second->gradient().size() / segment;

			float* mx = _m[param.first].data();
			float* px = param.second->params().data();
			float* gx = param.second->gradient().data();

			for (int i = 0; i < size; i++) {
				const __m256 gx256 = _mm256_load_ps(gx);
				__m256 px256 = _mm256_load_ps(px);
				__m256 mx256 = _mm256_load_ps(mx);

				mx256 = _mm256_add_ps(_mm256_mul_ps(beta1256, mx256), _mm256_mul_ps(ib1256, gx256));

				if (_t < 1e4)
				{
					const __m256 m_meanx256 = _mm256_div_ps(mx256, denb1256);
					px256 = _mm256_add_ps(px256, _mm256_mul_ps(alpha256, m_meanx256));
				}
				else
				{
					px256 = _mm256_add_ps(px256, _mm256_mul_ps(alpha256, mx256));
				}

				_mm256_storeu_ps(px, px256);
				_mm256_storeu_ps(mx, mx256);

				px += segment;
				gx += segment;
				mx += segment;
			}

			for (int i = size * segment; i < param.second->gradient().size(); i++) {
				*mx = _beta1 * *mx + ib1 * *gx;

				if (_t < 1e4)
				{
					const float m_meanx = *mx++ / denb1;

					*px++ += _alpha * m_meanx;
				}
				else
				{
					*px++ += _alpha * *mx++;
				}

				gx++;
			}
		}

	}
}
