#include "ContinuousExploration.h"
#include "RandomGenerator.h"

ContinuousExploration::ContinuousExploration(Coeus::IInterpolation* p_interpolation) :
	_method(),
	_interpolation(p_interpolation),
	_ounoise(nullptr),
	_sigma(0)
{
}

ContinuousExploration::~ContinuousExploration()
{
	delete _ounoise;
	delete _interpolation;
}

Tensor ContinuousExploration::explore(Tensor* p_action) const
{
	Tensor output(*p_action);
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

void ContinuousExploration::update(const int p_timestep)
{
	if (_interpolation != nullptr) _sigma = _interpolation->interpolate(p_timestep);

	if (_ounoise != nullptr)
	{
		_ounoise->set_sigma(_sigma);
	}
}

void ContinuousExploration::reset() const
{
	if (_ounoise != nullptr)
	{
		_ounoise->reset();
	}
}

void ContinuousExploration::init_gaussian(const float p_sigma)
{
	_method = GAUSSIAN;
	_sigma = p_sigma;
}

void ContinuousExploration::init_ounoise(const int p_dim, const float p_mu, const float p_sigma, const float p_theta)
{
	_method = OUNOISE;
	_sigma = p_sigma;
	_ounoise = new OUNoise(p_dim, p_mu, p_sigma, p_theta);
	
}

void ContinuousExploration::explore_gaussian(Tensor& p_action) const
{
	for (int i = 0; i < p_action.size(); i++)
	{
		const float rand = _sigma > 0 ? RandomGenerator::get_instance().normal_random(0, _sigma) : 0;
		p_action[i] += rand;
	}
}

void ContinuousExploration::explore_ounoise(Tensor& p_action) const
{
	_ounoise->noise(p_action);
}
