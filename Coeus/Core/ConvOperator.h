#pragma once
#include "NeuronOperator.h"

namespace Coeus
{
	class __declspec(dllexport) ConvOperator : public NeuronOperator
	{
	public:
		ConvOperator(int p_dim, ACTIVATION p_activation);
		ConvOperator(ConvOperator& p_copy, bool p_clone);
		~ConvOperator();

		void integrate(Tensor* p_dim_tensor, int p_rows, int p_cols, Tensor* p_input, Tensor* p_weights);
		void activate() override;

	private:
		void integrate(Tensor* p_input, Tensor* p_weights) override;
	};
}

