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

		void init_training(float p_alpha_a, float p_alpha_e, float p_alpha_i);

		float alpha_a() const { return _alpha_a; }
		float alpha_e() const { return _alpha_e; }
		float alpha_i() const { return _alpha_i; }

	private:
		float _alpha_a;
		float _alpha_e;
		float _alpha_i;
	};
}


