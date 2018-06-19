#pragma once
#include "BaseLayer.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayer : public BaseLayer
	{
		friend class LSTMLayerGradient;
	public:
		LSTMLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation);
		~LSTMLayer();

		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		void override_params(BaseLayer* p_source) override;
		void post_connection(BaseLayer* p_input) override;

	private:		
		NeuralGroup* _hf;
		NeuralGroup* _hi;
		NeuralGroup* _ho;
		NeuralGroup* _hc;
		Tensor*		 _x;
		Tensor*		 _h;
		Tensor*		 _h_old;
		Tensor*		 _c;
		Tensor*		 _c_old;

		Connection* _Wf;
		Connection* _Wi;
		Connection* _Wo;
		Connection* _Wc;
		Connection* _Wy;

	};
}


