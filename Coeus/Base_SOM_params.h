#pragma once
#include "SOM.h"

namespace Coeus {
	class __declspec(dllexport) Base_SOM_params
	{
	public:
		Base_SOM_params(SOM* p_som);
		virtual ~Base_SOM_params();

		virtual void param_decay();

		float sigma() const { return _sigma; }

	protected:
		void init(float p_epochs);

		float _sigma0;
		float _sigma;
		float _lambda;
		float _iteration;
		float _epochs;

	};
}


