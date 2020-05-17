#pragma once
#include "iinterpolation.h"
#include "tensor.h"

class COEUS_DLL_API discrete_exploration
{
public:
	enum METHOD
	{
		EGREEDY,
		BOLTZMAN
	};
	
	discrete_exploration(METHOD p_method, float p_exploration_parameter, iinterpolation* p_interpolation = nullptr);
	~discrete_exploration();

	tensor	explore(tensor& p_values) const;
	void	update(int p_timestep);

private:
	void explore_egreedy(tensor& p_output, tensor& p_values) const;
	void explore_boltzman(tensor& p_output, tensor& p_values) const;
	
	METHOD	_method;
	float	_param;

	iinterpolation* _interpolation;
};
