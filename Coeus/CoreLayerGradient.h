#pragma once
#include "IGradientComponent.h"

namespace Coeus {

	class __declspec(dllexport) CoreLayerGradient : public IGradientComponent
	{
	public:
		CoreLayerGradient();
		~CoreLayerGradient();

		void init(BaseLayer* p_layer) override;
		map<string, Tensor>* calc_delta(Tensor* p_delta) override;
		map<string, Tensor>* calc_gradient() override;

	};

}