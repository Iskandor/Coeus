#pragma once
#include <IInterpolation.h>
#include "Tensor.h"

class __declspec(dllexport) DiscreteExploration
{
public:
	enum METHOD
	{
		EGREEDY,
		BOLTZMAN
	};
	
	DiscreteExploration(METHOD p_method, float p_exploration_parameter, Coeus::IInterpolation* p_interpolation = nullptr);
	~DiscreteExploration();

	Tensor	explore(Tensor* p_values) const;
	void	update(int p_timestep);

private:
	void explore_egreedy(Tensor& p_output, Tensor* p_values) const;
	void explore_boltzman(Tensor& p_output, Tensor* p_values) const;
	
	METHOD	_method;
	float	_param;

	Coeus::IInterpolation* _interpolation;
};
