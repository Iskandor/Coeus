#pragma once
#include "BaseLayer.h"

namespace Coeus {

	class __declspec(dllexport) UBALLayer : public BaseLayer
	{
	public:

		UBALLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, BaseLayer* p_layer);
		~UBALLayer();

		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		void activate_back(Tensor* p_input = nullptr);
		void override(BaseLayer* p_source) override;

	private:
		NeuralGroup* _input_bp;
		NeuralGroup* _input_echo;
		NeuralGroup* _output_bp;
		NeuralGroup* _output_echo;

		Connection* _forward_W;
		Connection* _backward_M;
	};

}


