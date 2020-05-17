#pragma once
#include "tensor.h"

class COEUS_DLL_API ounoise {
public:
	ounoise(int p_dim, float p_mu = 0.f, float p_sigma = 0.2f, float p_theta = 0.15f, float p_dt = 0.01f);
	~ounoise();

	void reset();
	void noise(tensor& p_action) const;
	void set_sigma(float p_sigma);
	
private:
	int _dim;
	float _mu;
	float _theta;
	float _sigma;
	float _dt;
	tensor _state;
};
