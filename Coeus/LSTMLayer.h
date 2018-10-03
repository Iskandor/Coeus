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

	private:
		LSTMCellGroup*	_memory_cells;
		SimpleCellGroup*	_input_gate;
		SimpleCellGroup*	_output_gate;
	};
}


