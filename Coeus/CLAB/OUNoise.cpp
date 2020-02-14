#include "OUNoise.h"
#include "RandomGenerator.h"

OUNoise::OUNoise(const int p_dim, const float p_mu, const float p_sigma, const float p_theta) : _dim(p_dim), _mu(p_mu), _theta(p_theta), _sigma(p_sigma)
{
	_state = Tensor({ p_dim }, Tensor::VALUE, p_mu);
}

OUNoise::~OUNoise()
= default;

void OUNoise::reset()
{
	_state = Tensor({ _dim }, Tensor::VALUE, _mu);
}

void OUNoise::noise(Tensor& p_action) const
{
	for(int i = 0; i < _dim; i++)
	{
		_state[i] += _theta * (_mu - _state[i]) + _sigma * RandomGenerator::get_instance().normal_random();
		p_action[i] += _state[i];
	}
}

void OUNoise::set_sigma(const float p_sigma)
{
	_sigma = p_sigma;
}
