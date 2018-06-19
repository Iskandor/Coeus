#pragma once
#include "IGradientComponent.h"
#include "LSTMLayer.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayerGradient : public IGradientComponent
	{
	public:
		LSTMLayerGradient(LSTMLayer* p_layer);
		~LSTMLayerGradient();

		void set_delta(Tensor* p_delta) override;
		void init() override;
		void calc_deriv() override;
		void calc_delta(Tensor* p_weights, LayerState* p_state) override;
		void update(map<string, Tensor> &p_update) override;
		void calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) override;

	private:
		Tensor	_dc_next;
		Tensor	_dh_next;
	};
}
