#pragma once
#include "BaseLayer.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayer : public BaseLayer
	{
		friend class LSTMLayerGradient;
	public:
		LSTMLayer(string p_id, int p_dim, ACTIVATION p_activation);
		~LSTMLayer();

		void init(vector<BaseLayer*>& p_input_layers) override;
		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		void override(BaseLayer* p_source) override;

	private:		
		NeuralGroup* _hf;
		NeuralGroup* _hi;
		NeuralGroup* _ho;
		NeuralGroup* _hc;
		NeuralGroup* _x;
		Tensor*		 _h;
		Tensor*		 _h_old;
		Tensor*		 _c;
		Tensor*		 _c_old;
		Tensor		 _input_buffer;

		Connection* _Wf;
		Connection* _Wi;
		Connection* _Wo;
		Connection* _Wc;
		Connection* _Wy;

	};
}


