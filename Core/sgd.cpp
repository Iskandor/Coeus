#include "sgd.h"
#include "CLAB.h"


sgd::sgd(neural_network* p_model, const float p_alpha, const float p_momentum, const bool p_nesterov, float p_weight_decay) : optimizer(p_model, p_alpha, p_weight_decay),
	_momentum(p_momentum),
	_nesterov(p_nesterov)
{
	if (_momentum > 0)
	{
		_velocity = p_model->zero_model();
	}	
}


sgd::~sgd()
= default;

void sgd::update()
{	
	optimizer::update();

	const __m256 alpha256 = _mm256_broadcast_ss(&_alpha);

	if (_momentum > 0)
	{
		const __m256 momentum256 = _mm256_broadcast_ss(&_momentum);

		if (_nesterov) {
			// x += -mu * v + (1 + mu) * mu * v - learning_rate * dx
			float minus_momentum = -_momentum;
			float one_plus_momentum = 1 + _momentum;
			const __m256 minus_momentum256 = _mm256_broadcast_ss(&minus_momentum);
			const __m256 one_plus_momentum256 = _mm256_broadcast_ss(&one_plus_momentum);

			for (auto& param : *_model)
			{
				const int size = param.second->gradient().size() / segment;
				float* px = param.second->params().data();
				float* gx = param.second->gradient().data();
				float* vx = _velocity[param.first].data();

				for (int i = 0; i < size; i++) {
					const __m256 gx256 = _mm256_load_ps(gx);
					__m256 px256 = _mm256_load_ps(px);
					__m256 vx256 = _mm256_load_ps(vx);

					__m256 v256 = _mm256_add_ps(_mm256_mul_ps(vx256, momentum256), _mm256_mul_ps(gx256, alpha256));
					px256 = _mm256_sub_ps(px256, _mm256_add_ps(_mm256_mul_ps(minus_momentum256, vx256), _mm256_mul_ps(one_plus_momentum256, v256)));
					vx256 = v256;

					_mm256_storeu_ps(px, px256);
					_mm256_storeu_ps(vx, vx256);
					px += segment;
					gx += segment;
					vx += segment;
				}

				float v = 0;

				for (int i = size * segment; i < param.second->gradient().size(); i++) {
					v = _momentum * *vx + _alpha * *gx++;
					*px++ -= -_momentum * *vx + (1 + _momentum) * v;
					*vx++ = v;
				}
			}
		}
		else
		{
			// v = mu * v - learning_rate * dx
			for (auto& param : *_model)
			{
				const int size = param.second->gradient().size() / segment;

				float* px = param.second->params().data();
				float* gx = param.second->gradient().data();
				float* vx = _velocity[param.first].data();

				for (int i = 0; i < size; i++) {
					const __m256 gx256 = _mm256_load_ps(gx);
					__m256 px256 = _mm256_load_ps(px);					
					__m256 vx256 = _mm256_load_ps(vx);

					vx256 = _mm256_add_ps(_mm256_mul_ps(vx256, momentum256), _mm256_mul_ps(gx256, alpha256));
					px256 = _mm256_sub_ps(px256, vx256);

					_mm256_storeu_ps(px, px256);
					_mm256_storeu_ps(vx, vx256);
					px += segment;
					gx += segment;
					vx += segment;
				}

				for (int i = size * segment; i < param.second->gradient().size(); i++) {
					*vx = _momentum * *vx + _alpha * *gx++;
					*px++ -= *vx++;
				}
			}
		}
	}
	else
	{
		for (auto& param : *_model)
		{
			const int size = param.second->gradient().size() / segment;

			float* px = param.second->params().data();
			float* gx = param.second->gradient().data();

			for (int i = 0; i < size; i++) {
				const __m256 gx256 = _mm256_load_ps(gx);
				__m256 px256 = _mm256_load_ps(px);

				px256 = _mm256_sub_ps(px256, _mm256_mul_ps(gx256, alpha256));

				_mm256_storeu_ps(px, px256);
				px += segment;
				gx += segment;
			}

			for (int i = size * segment; i < param.second->gradient().size(); i++) {
				*px++ -= *gx++ * _alpha;
			}
		}
	}
}


