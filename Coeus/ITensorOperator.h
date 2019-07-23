#pragma once
class __declspec(dllexport) ITensorOperator
{
public:
	ITensorOperator(const ITensorOperator&) = delete;
	ITensorOperator& operator=(const ITensorOperator &) = delete;
	ITensorOperator(ITensorOperator &&) = delete;
	ITensorOperator& operator=(ITensorOperator &&) = delete;

	virtual void vv_add(float* p_x, float* p_y, float* p_z, int p_size) = 0;
	virtual void vv_sub(float* p_x, float* p_y, float* p_z, int p_size) = 0;
	virtual void vv_ewprod(float* p_x, float* p_y, float* p_z, int p_size) = 0;
	virtual void vv_ewdiv(float* p_x, float* p_y, float* p_z, int p_size) = 0;
	virtual void vc_prod(float* p_x, float p_y, float* p_z, int p_size) = 0;
	virtual void vc_prod_add(float* p_x, float p_y, float* p_z, int p_size) = 0;

	virtual void Mv_add(float* p_A, float* p_x, float* p_B, int p_rows, int p_cols) = 0;

	virtual void vM_prod(float* p_x, float* p_A, float* p_y, int p_rows, int p_cols) = 0;
	virtual void MM_prod(float* p_A, bool p_Atrans, float* p_B, bool p_Btrans, float* p_C, int p_rows, int p_common, int p_cols) = 0;

	virtual void v_reduce(float* p_x, float* p_y, int p_size) = 0;
	virtual void M_reduce(float* p_x, float* p_A, bool p_row_major, int p_rows, int p_cols) = 0;
	virtual void V_reduce(float* p_A, float* p_V, int p_batch, int p_rows, int p_cols) = 0;

	virtual void full_int_s(float* p_net, float* p_x, float* p_w, int p_rows, int p_cols) = 0;
	virtual void full_int_b(int p_batch, float* p_net, float* p_x, float* p_w, int p_rows, int p_cols) = 0;
	virtual void full_bias_s(float* p_net, float* p_bias, int p_rows) = 0;
	virtual void full_bias_b(int p_batch, float* p_net, float* p_bias, int p_rows) = 0;
	virtual void full_delta(int p_batch, float* p_delta0, float* p_delta1, float* p_w, int p_rows, int p_cols) = 0;
	virtual void full_w_gradient(int p_batch, float* p_x0, float* p_delta1, float* p_grad, int p_rows, int p_cols) = 0;
	virtual void full_b_gradient(int p_batch, float* p_delta1, float* p_grad, int p_rows) = 0;

	virtual void lstm_state_s(float* p_state, float* p_ig, float* p_fg, float* p_cec, int p_size) = 0;
	virtual void lstm_state_b(int p_batch, float* p_state, float* p_ig, float* p_fg, float* p_cec, int p_size) = 0;
	virtual void lstm_delta_s(float* p_delta0, float* p_derivative, float* p_output, float* p_delta1, int p_size) = 0;
	virtual void lstm_delta_b(int p_batch, float* p_delta0, float* p_derivative, float* p_output, float* p_delta1, int p_size) = 0;
	virtual void lstm_derivative_s(float* p_derivative, float* p_fg, float* p_arg1, float* p_arg2, float* p_input, int p_rows, int p_cols) = 0;
	virtual void lstm_derivative_b(int p_batch, float* p_derivative, float* p_fg, float* p_arg1, float* p_arg2, float* p_input, int p_rows, int p_cols) = 0;
	virtual void lstm_gradient_s(float* p_gradient, float* p_error, float* p_derivative, int p_rows, int p_cols) = 0;
	virtual void lstm_gradient_b(int p_batch, float* p_gradient, float* p_error, float* p_derivative, int p_rows, int p_cols) = 0;

protected:
	ITensorOperator() = default;
	~ITensorOperator() = default;
};
