#pragma once
#include "Tensor.h"
#include "IActivationFunction.h"
#include "BaseCellGroup.h"
#include "SimpleCellGroup.h"

using namespace FLAB;

namespace Coeus {
	class __declspec(dllexport) LSTMCellGroup : public BaseCellGroup
	{
	public:
		LSTMCellGroup(int p_dim, ACTIVATION p_activation_function, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate);
		explicit LSTMCellGroup(nlohmann::json p_data);
		LSTMCellGroup(LSTMCellGroup& p_copy);
		LSTMCellGroup& operator = (const LSTMCellGroup& p_copy);

		~LSTMCellGroup();

		void integrate(Tensor* p_input, Tensor* p_weights) override;
		void activate() override;

		LSTMCellGroup* clone() override;

	private:
		void activate(Tensor* p_input_gate, Tensor* p_output_gate);
		Tensor	_state;

		IActivationFunction* _g;

		SimpleCellGroup* _input_gate;
		SimpleCellGroup* _output_gate;
	};
}
