#pragma once
#include "IGradientComponent.h"
#include "CoreLayer.h"

namespace Coeus {

	class __declspec(dllexport) CoreLayerGradient : public IGradientComponent
	{
	public:
		CoreLayerGradient(BaseLayer* p_layer, NeuralNetwork* p_network);
		~CoreLayerGradient();

		void init() override;
		void calc_deriv() override;
		void calc_delta(Tensor* p_weights, Tensor* p_delta) override;
		void calc_gradient(map<string, Tensor> &p_w_gradient, map<string, Tensor> &p_b_gradient) override;
	};

}
