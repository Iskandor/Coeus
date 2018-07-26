#pragma once
#include "IGradientComponent.h"
#include "RecurrentLayer.h"

namespace Coeus
{
	class __declspec(dllexport) RecurrentLayerGradient : public IGradientComponent
	{
	public:
		explicit RecurrentLayerGradient(RecurrentLayer* p_recurrent_layer);
		~RecurrentLayerGradient();

		void init() override;
		void calc_deriv() override;
		void calc_delta(Tensor* p_weights, LayerState* p_state) override;
		void calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) override;
		void set_delta(Tensor* p_delta) override;
	};
}


