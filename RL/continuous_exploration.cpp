#include "continuous_exploration.h"
#include "random_generator.h"

continuous_exploration::continuous_exploration(iinterpolation* p_interpolation) :
	_method(),
	_interpolation(p_interpolation),
	_ounoise(nullptr),
	_sigma(0)
{
}

continuous_exploration::~continuous_exploration()
{
	delete _ounoise;
	delete _interpolation;
}

tensor continuous_exploration::explore(tensor& p_action) const
{
	tensor output(p_action);
	switch(_method)
	{
	case GAUSSIAN:
		explore_gaussian(output);
		break;
	case OUNOISE:
		explore_ounoise(output);
		break;
	default: ;
	}

	return output;
}

void continuous_exploration::update(const int p_timestep)
{
	if (_interpolation != nullptr) _sigma = _interpolation->interpolate(p_timestep);

	if (_ounoise != nullptr)
	{
		_ounoise->set_sigma(_sigma);
	}
}

void continuous_exploration::reset() const
{
	if (_ounoise != nullptr)
	{
		_ounoise->reset();
	}
}

void continuous_exploration::init_gaussian(const float p_sigma)
{
	_method = GAUSSIAN;
	_sigma = p_sigma;
}

void continuous_exploration::init_ounoise(const int p_dim, const float p_mu, const float p_sigma, const float p_theta)
{
	_method = OUNOISE;
	_sigma = p_sigma;
	_ounoise = new ounoise(p_dim, p_mu, p_sigma, p_theta);
	
}

void continuous_exploration::explore_gaussian(tensor& p_action) const
{
	for (int i = 0; i < p_action.size(); i++)
	{
		const float rand = _sigma > 0 ? random_generator::instance().normal_random(0, _sigma) : 0;
		p_action[i] += rand;
	}
}

void continuous_exploration::explore_ounoise(tensor& p_action) const
{
	_ounoise->noise(p_action);
}
