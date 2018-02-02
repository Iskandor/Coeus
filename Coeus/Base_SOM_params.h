#pragma once
#include "SOM.h"

namespace Coeus {
	class __declspec(dllexport) Base_SOM_params
	{
	public:
		Base_SOM_params(SOM* p_som);
		virtual ~Base_SOM_params();

		virtual void param_decay();

		double sigma() const { return _sigma; }

	protected:
		void init(double p_epochs);

		double _sigma0;
		double _sigma;
		double _lambda;
		double _iteration;

	};
}


