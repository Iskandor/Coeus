#pragma once

#include "Tensor.h"
#include "BaseLayer.h"

using namespace FLAB;

namespace Coeus {

	class __declspec(dllexport) IGradientComponent
	{
	public:
		IGradientComponent();
		~IGradientComponent();

		virtual void init(BaseLayer* p_layer) = 0;
		virtual map<string, Tensor>* calc_delta(Tensor* p_delta) = 0;
		virtual map<string, Tensor>* calc_gradient() = 0;

	private:
		map<string, Tensor> _deriv;
		map<string, Tensor> _delta;
		map<string, Tensor> _gradient;
	};

}


