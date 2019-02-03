#pragma once
#include "IGradientComponent.h"
#include "LSTMLayer.h"

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
		void calc_gradient(map<string, Tensor> &p_gradient) override;
		void calc_partial_deriv() override;
		void reset() override;

	private:
		map<string, Tensor> _partial_deriv;
		Tensor _state_error;
	};
}
