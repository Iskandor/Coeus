#pragma once
#include "Tensor.h"
#include "IInterpolation.h"
#include "OUNoise.h"

class __declspec(dllexport) ContinuousExploration
{
public:
	enum METHOD
	{
		GAUSSIAN,
		OUNOISE
	};

	ContinuousExploration(Coeus::IInterpolation* p_interpolation = nullptr);
	~ContinuousExploration();

	Tensor	explore(Tensor* p_action) const;
	void	update(int p_timestep);
	void	reset() const;

	void init_gaussian(float p_sigma = 1.0f);
	void init_ounoise(int p_dim, float p_mu = 0.f, float p_sigma = 0.2f, float p_theta = 0.15f);

private:
	void explore_gaussian(Tensor& p_action) const;
	void explore_ounoise(Tensor& p_action) const;

	METHOD	_method;

	Coeus::IInterpolation* _interpolation;
	OUNoise* _ounoise;

	float _sigma;
};
