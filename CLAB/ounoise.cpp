#include "ounoise.h"
#include "random_generator.h"

/**
 * \brief Ornstein–Uhlenbeck process
 * \param p_dim state dimension
 * \param p_mu drift constant
 * \param p_sigma std. deviation of gaussian
 * \param p_theta 
 * \param p_dt 
 */
ounoise::ounoise(const int p_dim, const float p_mu, const float p_sigma, const float p_theta, const float p_dt) : _dim(p_dim), _mu(p_mu), _theta(p_theta), _sigma(p_sigma), _dt(p_dt)
{
	_state = tensor({ p_dim }, tensor::VALUE, p_mu);
}

ounoise::~ounoise()
= default;

/**
 * \brief Reset state to mu value
 */
void ounoise::reset()
{
	_state = tensor({ _dim }, tensor::VALUE, _mu);
}

/**
 * \brief Add noise to input action tensor
 * \param p_action action tensor which is modified inside method
 */
void ounoise::noise(tensor& p_action) const
{
	for(int i = 0; i < _dim; i++)
	{
		_state[i] += _theta * (_mu - _state[i]) * _dt + _sigma * random_generator::instance().normal_random() * sqrt(_dt);
		p_action[i] += _state[i];
	}
}

/**
 * \brief Set new value for std. deviation
 * \param p_sigma std. deviation
 */
void ounoise::set_sigma(const float p_sigma)
{
	_sigma = p_sigma;
}
