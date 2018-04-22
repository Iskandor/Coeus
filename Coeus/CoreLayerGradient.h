#pragma once
#include "IGradientComponent.h"
#include "CoreLayer.h"

namespace Coeus {

	class __declspec(dllexport) CoreLayerGradient : public IGradientComponent
	{
	public:
		explicit CoreLayerGradient(CoreLayer* p_layer);
		~CoreLayerGradient();

		void init() override;
		void calc_deriv() override;
		void calc_delta(Tensor* p_weights, Tensor* p_delta) override;
		void calc_gradient(map<string, Tensor> &p_gradient) override;

	};

}
