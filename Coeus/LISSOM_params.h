#pragma once
#include "Base_SOM_params.h"
#include "LISSOM.h"

namespace Coeus
{
	class __declspec(dllexport) LISSOM_params : public Base_SOM_params
	{
	public:
		explicit LISSOM_params(LISSOM* p_lissom);
		~LISSOM_params();

		void init_training(double p_alpha_a, double p_alpha_e, double p_alpha_i);

		double alpha_a() const { return _alpha_a; }
		double alpha_e() const { return _alpha_e; }
		double alpha_i() const { return _alpha_i; }

	private:
		double _alpha_a;
		double _alpha_e;
		double _alpha_i;
	};
}


