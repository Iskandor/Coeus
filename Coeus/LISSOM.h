#pragma once
#include "SOM.h"

namespace Coeus {

	class __declspec(dllexport) LISSOM : public SOM
	{
	public:
		LISSOM(string p_id, const int p_input_dim, const int p_dim_x, const int p_dim_y, const ACTIVATION p_activation, float p_gamma_e, float p_gamma_i);
		~LISSOM();

		void activate(Tensor* p_input = nullptr) override;

		Connection* get_latteral_e() const { return _lateral_e;}
		Connection* get_latteral_i() const { return _lateral_i; }

	private:
		Tensor		_auxoutput;
		Tensor		_prime_activity;

		Connection* _lateral_e;
		Connection* _lateral_i;

		float _gamma_e;
		float _gamma_i;
	};
}

