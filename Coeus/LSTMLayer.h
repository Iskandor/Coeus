#pragma once
#include "BaseLayer.h"
#include "LSTMCellGroup.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayer : public BaseLayer
	{
		friend class LSTMLayerGradient;
	public:
		LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation);
		~LSTMLayer();

		void init(vector<BaseLayer*>& p_input_layers) override;
		void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
		void activate(Tensor* p_input = nullptr) override;
		void override(BaseLayer* p_source) override;
		void reset() override;

	private:
		LSTMCellGroup*		_cec;
		SimpleCellGroup*	_input_gate;
		SimpleCellGroup*	_output_gate;
		SimpleCellGroup*	_forget_gate;
		SimpleCellGroup*	_aux_input;
		SimpleCellGroup*	_context;

		Connection* _in_input_gate;
		Connection* _in_output_gate;
		Connection* _in_forget_gate;
		Connection* _ct_cec;

		map<string, Tensor> _partial_deriv;

		vector<Tensor*> _input;
	};
}


