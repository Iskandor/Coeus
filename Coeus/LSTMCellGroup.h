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
		LSTMCellGroup(int p_dim, ACTIVATION p_activation_function, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup*p_forget_gate);
		LSTMCellGroup(json p_data, SimpleCellGroup* p_input_gate, SimpleCellGroup* p_output_gate, SimpleCellGroup*p_forget_gate);
		LSTMCellGroup(LSTMCellGroup& p_copy);
		LSTMCellGroup& operator = (const LSTMCellGroup& p_copy);

		~LSTMCellGroup();

		void integrate(Tensor* p_input, Tensor* p_weights) override;
		void activate() override;
		void reset() const;

		LSTMCellGroup* clone() override;

		Tensor get_h() const;
		Tensor get_dh();
		Tensor get_g() const;
		Tensor get_dg();

		json get_json() const override;

	private:
		void activate(Tensor* p_input_gate, Tensor* p_output_gate, Tensor* p_forget_gate);
		Tensor _state;
		Tensor _h_output;
		Tensor _g_output;

		IActivationFunction* _g;

		SimpleCellGroup* _input_gate;
		SimpleCellGroup* _output_gate;
		SimpleCellGroup* _forget_gate;
	};
}
