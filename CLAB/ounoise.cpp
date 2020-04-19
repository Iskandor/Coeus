#include "ounoise.h"
#include "random_generator.h"

ounoise::ounoise(const int p_dim, const float p_mu, const float p_sigma, const float p_theta) : _dim(p_dim), _mu(p_mu), _theta(p_theta), _sigma(p_sigma)
{
	_state = tensor({ p_dim }, tensor::VALUE, p_mu);
}

ounoise::~ounoise()
= default;

void ounoise::reset()
{
	_state = tensor({ _dim }, tensor::VALUE, _mu);
}

void ounoise::noise(tensor& p_action) const
{
	for(int i = 0; i < _dim; i++)
	{
		_state[i] += _theta * (_mu - _state[i]) + _sigma * random_generator::instance().normal_random() * sqrt(1e-2f);
		p_action[i] += _state[i];
	}
}

void ounoise::set_sigma(const float p_sigma)
{
	_sigma = p_sigma;
}
