#pragma once
#include "ITensorOperator.h"

namespace Coeus
{
	class __declspec(dllexport) TensorOperatorMKL : public ITensorOperator
	{
	public:
		TensorOperatorMKL() = default;
		virtual ~TensorOperatorMKL() = default;

		void vv_add(float* p_x, float* p_y, float* p_z, int p_size) override;
		void vv_sub(float* p_x, float* p_y, float* p_z, int p_size) override;
		void vv_ewprod(float* p_x, float* p_y, float* p_z, int p_size) override;
		void vv_ewdiv(float* p_x, float* p_y, float* p_z, int p_size) override;
		void vc_prod(float* p_x, float p_y, float* p_z, int p_size) override;
		void vc_prod_add(float* p_x, float p_y, float* p_z, int p_size) override;

		void vM_prod(float* p_x, float* p_A, float* p_y, int p_rows, int p_cols) override;

		void full_int_s(float* p_net, float* p_x, float* p_w, int p_rows, int p_cols) override;
		void full_int_b(int p_batch, float* p_net, float* p_x, float* p_w, int p_rows, int p_cols) override;
		void full_bias_s(float* p_net, float* p_bias, int p_rows) override;
		void full_bias_b(int p_batch, float* p_net, float* p_bias, int p_rows) override;

		void lstm_state_s(float* p_state, float* p_ig, float* p_fg, float* p_cec, int p_size) override;
		void lstm_state_b(int p_batch, float* p_state, float* p_ig, float* p_fg, float* p_cec, int p_size) override;

		void full_delta_s(float* p_delta0, float* p_delta1, float* p_w, float* p_derivative, int p_rows, int p_cols) override;
		void full_delta_b(int p_batch, float* p_delta0, float* p_delta1, float* p_w, float* p_derivative, int p_rows, int p_cols) override;
		void full_gradient_s(float* p_x0, float* p_delta1, float* p_grad, int p_rows, int p_cols) override;
		void full_gradient_b(int p_batch, float* p_x0, float* p_delta1, float* p_grad, int p_rows, int p_cols) override;

		void lstm_delta_s(float* p_delta0, float* p_derivative, float* p_output, float* p_delta1, int p_size) override;
		void lstm_delta_b(int p_batch, float* p_delta0, float* p_derivative, float* p_output, float* p_delta1, int p_size) override;
		void lstm_derivative_s(float* p_derivative, float* p_fg, float* p_arg1, float* p_arg2, float* p_input, int p_rows, int p_cols) override;
		void lstm_derivative_b(int p_batch, float* p_derivative, float* p_fg, float* p_arg1, float* p_arg2, float* p_input, int p_rows, int p_cols) override;
		void lstm_gradient_s(float* p_gradient, float* p_error, float* p_derivative, int p_rows, int p_cols) override;
		void lstm_gradient_b(int p_batch, float* p_gradient, float* p_error, float* p_derivative, int p_rows, int p_cols) override;

		void m_reduce(float* p_x, float* p_A, int p_rows, int p_cols) override;
		void V_reduce(float* p_A, float* p_V, int p_batch, int p_rows, int p_cols) override;
		
	};
}