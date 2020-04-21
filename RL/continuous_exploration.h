#pragma once
#include "tensor.h"
#include "iinterpolation.h"
#include "ounoise.h"

class __declspec(dllexport) continuous_exploration
{
public:
	enum METHOD
	{
		GAUSSIAN,
		OUNOISE
	};

	continuous_exploration(iinterpolation* p_interpolation = nullptr);
	~continuous_exploration();

	tensor	explore(tensor& p_action) const;
	void	update(int p_timestep);
	void	reset() const;

	void init_gaussian(float p_sigma = 1.0f);
	void init_ounoise(int p_dim, float p_mu = 0.f, float p_sigma = 0.2f, float p_theta = 0.15f);

private:
	void explore_gaussian(tensor& p_action) const;
	void explore_ounoise(tensor& p_action) const;

	METHOD	_method;

	iinterpolation* _interpolation;
	ounoise* _ounoise;

	float _sigma;
};
