#pragma once
#include "IGradientComponent.h"
#include "LSTMLayer.h"
#include "LSTMLayerState.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayerGradient : public IGradientComponent
	{
	public:
		LSTMLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network);
		~LSTMLayerGradient();

		void init() override;
		void calc_deriv() override;
		void calc_delta(Tensor* p_weights, Tensor* p_delta) override;
		void calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) override;

	private:
		Tensor _state_error;
	};
}
