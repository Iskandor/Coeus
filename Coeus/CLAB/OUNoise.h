#pragma once
#include "Tensor.h"

class __declspec(dllexport) OUNoise {
public:
	OUNoise(int p_dim, float p_mu = 0.f, float p_sigma = 0.2f, float p_theta = 0.15f);
	~OUNoise();

	void reset();
	void noise(Tensor& p_action) const;
	void set_sigma(float p_sigma);
	
private:
	int _dim;
	float _mu;
	float _theta;
	float _sigma;
	Tensor _state;
};
