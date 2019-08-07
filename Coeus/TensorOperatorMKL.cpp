#include "TensorOperatorMKL.h"
#include <mkl_cblas.h>
#include <mkl.h>
#include <cstring>
#include "Tensor.h"
#include <iostream>

using namespace Coeus;

void TensorOperatorMKL::full_int_s(float* p_net, float* p_x, float* p_w, const int p_rows, const int p_cols)
{
	memset(p_net, 0, sizeof(float) * p_rows);
	cblas_sgemv(CblasRowMajor, CblasNoTrans, p_rows, p_cols, 1, p_w, p_cols, p_x, 1, 0, p_net, 1);
}

void TensorOperatorMKL::full_int_b(const int p_batch, float* p_net, float* p_x, float* p_w, const int p_rows, const int p_cols)
{
	MM_prod(p_x, false, p_w, true, p_net, p_batch, p_cols, p_rows, false);
}

void TensorOperatorMKL::full_bias_s(float* p_net, float* p_bias, const int p_rows)
{
	if (p_bias != nullptr)
	{
		vsAdd(p_rows, p_net, p_bias, p_net);
	}
}

void TensorOperatorMKL::full_bias_b(const int p_batch, float* p_net, float* p_bias, const int p_rows)
{
	if (p_bias != nullptr)
	{
		float* nx = &p_net[0];

		for (int j = 0; j < p_batch; j++)
		{
			float* bx = &p_bias[0];
			for (int i = 0; i < p_rows; i++)
			{
				*nx++ += *bx++;
			}
		}
	}
}

void TensorOperatorMKL::lstm_state(const int p_batch, float* p_state, float* p_ig, float* p_fg, float* p_cec, const int p_size)
{
	float* sx = &p_state[0];
	float* cx = &p_cec[0];
	float* igx = &p_ig[0];
	float* fgx = &p_fg[0];

	for(int i = 0; i < p_batch * p_size; i++)
	{
		*sx = *fgx++ * *sx + *igx++ * *cx++;
		sx++;
	}
}

void TensorOperatorMKL::lstm_delta(const int p_batch, float* p_delta0, float* p_derivative, float* p_output, float* p_delta1, const int p_size)
{
	float* d0x = &p_delta0[0];
	float* dx = &p_derivative[0];
	float* ox = &p_output[0];
	float* d1x = &p_delta1[0];

	for (int i = 0; i < p_batch * p_size; i++)
	{
		*d0x++ = *dx++ * *ox++ * *d1x++;
	}
}

void TensorOperatorMKL::lstm_derivative(const int p_batch, float* p_derivative, float* p_fg, float* p_arg1, float* p_arg2, float* p_input, const int p_rows, const int p_cols)
{
	float* gx = &p_derivative[0];
	float* fgx = &p_fg[0];
	float* a1x = &p_arg1[0];
	float* a2x = &p_arg2[0];

	for (int b = 0; b < p_batch; b++)
	{
		for (int i = 0; i < p_rows; i++)
		{
			float* x = &p_input[b * p_cols];
			const float a = *a1x * *a2x;

			for (int j = 0; j < p_cols; j++)
			{
				*gx = *gx * *fgx + a * *x++;
				gx++;
			}

			fgx++;
			a1x++;
			a2x++;
		}
	}
}

void TensorOperatorMKL::lstm_w_gradient(int p_batch, float* p_gradient, float* p_error, float* p_derivative, const int p_rows, const int p_cols)
{
	memset(p_gradient, 0, sizeof(float) * p_cols * p_rows);

	float* ex = &p_error[0];
	float* dx = &p_derivative[0];

	for (int b = 0; b < p_batch; b++)
	{
		float *gx = &p_gradient[0];

		for (int i = 0; i < p_rows; i++)
		{
			//cblas_saxpy(p_cols, *ex, (dx + b * p_rows * p_cols + i * p_cols), 1, (gx + i * p_cols), 1);
			for (int j = 0; j < p_cols; j++)
			{
				*gx++ += *ex * *dx++;
			}
			ex++;
		}
	}
}

void TensorOperatorMKL::gru_state(int p_batch, float* p_state, float* p_zg, float* p_hcan, int p_size)
{
	float* hx = &p_state[0];
	float* zgx = &p_zg[0];
	float* hcx = &p_hcan[0];

	for (int i = 0; i < p_batch * p_size; i++)
	{
		*hx = *zgx * *hx++ + (1 - *zgx++) * *hcx++;
	}
}

void TensorOperatorMKL::full_delta(const int p_batch, float* p_delta0, float* p_delta1, float* p_w, const int p_rows, const int p_cols)
{
	MM_prod(p_delta1, false, p_w, false, p_delta0, p_batch, p_rows, p_cols, false);
}

void TensorOperatorMKL::full_w_gradient(const int p_batch, float* p_x0, float* p_delta1, float* p_grad, const int p_rows, const int p_cols, bool p_accumulate)
{
	MM_prod(p_delta1, true, p_x0, false, p_grad, p_rows, p_batch, p_cols, p_accumulate);
}

void TensorOperatorMKL::full_b_gradient(const int p_batch, float* p_delta1, float* p_grad, const int p_rows, bool p_accumulate)
{
	if (p_batch == 1)
	{
		if (p_accumulate)
		{
			vv_add(p_delta1, p_grad, p_grad, p_rows);
		}
		else
		{
			memcpy(p_grad, p_delta1, sizeof(float) * p_rows);
		}		
	}
	if (p_batch > 1)
	{
		M_reduce(p_grad, p_delta1, false, p_batch, p_rows, p_accumulate);
	}
}

void TensorOperatorMKL::v_reduce(float* p_x, float* p_y, const int p_size)
{
	p_x[0] = 0;
	float* y = &p_y[0];

	for (int i = 0; i < p_size; i++)
	{
		p_x[0] += *y++;
	}
}

void TensorOperatorMKL::M_reduce(float* p_x, float* p_A, bool p_row_major, const int p_rows, const int p_cols, bool p_accumulate)
{
	float* x = &p_x[0];

	if (p_row_major)
	{	
		float *a = &p_A[0];
		for (int i = 0; i < p_rows; i++)
		{
			if (!p_accumulate) *x = 0;
			for (int j = 0; j < p_cols; j++)
			{
				*x += *a++;
			}
			x++;
		}
	}
	else
	{
		for (int j = 0; j < p_cols; j++)
		{
			if (!p_accumulate) *x = 0;

			for (int i = 0; i < p_rows; i++)
			{
				*x += p_A[i * p_cols + j];
			}
			x++;
		}
	}
}

void TensorOperatorMKL::V_reduce(float* p_A, float* p_V, const int p_batch, const int p_rows, const int p_cols)
{
	float *vx = &p_V[0];

	memset(p_A, 0, sizeof(int) * p_cols * p_rows);

	for (int i = 0; i < p_batch; i++)
	{
		float *ax = &p_A[0];

		for (int j = 0; j < p_rows; j++)
		{
			for (int k = 0; k < p_cols; k++)
			{
				*ax++ += *vx++;
			}
		}
	}
}

void TensorOperatorMKL::vv_add(float* p_x, float* p_y, float* p_z, const int p_size)
{
	vsAdd(p_size, p_x, p_y, p_z);
}

void TensorOperatorMKL::vv_sub(float* p_x, float* p_y, float* p_z, int p_size)
{
	vsSub(p_size, p_x, p_y, p_z);
}

void TensorOperatorMKL::vv_ewprod(float* p_x, float* p_y, float* p_z, const int p_size)
{
	vsMul(p_size, p_x, p_y, p_z);
}

void TensorOperatorMKL::vv_ewdiv(float* p_x, float* p_y, float* p_z, const int p_size)
{
	float* x = &p_x[0];
	float* y = &p_y[0];
	float* z = &p_z[0];

	for (int i = 0; i < p_size; i++)
	{
		*z++ = *x++ / *y++;
	}
}

void TensorOperatorMKL::vc_prod(float* p_x, const float p_y, float* p_z, const int p_size)
{
	float* x = &p_x[0];
	float* z = &p_z[0];

	for (int i = 0; i < p_size; i++)
	{
		*z++ = *x++ * p_y;
	}
}

void TensorOperatorMKL::vc_prod_add(float* p_x, const float p_y, float* p_z, const int p_size)
{
	cblas_saxpy(p_size, p_y, p_x, 1, p_z, 1);
}

void TensorOperatorMKL::Mv_add(float* p_A, float* p_x, float* p_B, int p_rows, int p_cols)
{
	float* a = &p_A[0];
	float* x = &p_x[0];
	float* b = &p_B[0];

	for(int i = 0; i < p_rows; i++)
	{
		for(int j = 0; j < p_cols; j++)
		{
			*b++ = *a++ + *x;
		}
		x++;
	}
}

void TensorOperatorMKL::vM_prod(float* p_x, float* p_A, float* p_y, int p_rows, int p_cols)
{
	memset(p_y, 0, sizeof(float) * p_rows);
	cblas_sgemv(CblasRowMajor, CblasNoTrans, p_rows, p_cols, 1, p_A, p_cols, p_x, 1, 0, p_y, 1);
}

void TensorOperatorMKL::MM_prod(float* p_A, bool p_Atrans, float* p_B, bool p_Btrans, float* p_C, int p_rows, int p_common, int p_cols, bool p_accumulate)
{
	const CBLAS_TRANSPOSE Atrans = p_Atrans ? CblasTrans : CblasNoTrans;
	const CBLAS_TRANSPOSE Btrans = p_Btrans ? CblasTrans : CblasNoTrans;

	const int lda = p_Atrans ? p_rows : p_common;
	const int ldb = p_Btrans ? p_common : p_cols;

	cblas_sgemm(CblasRowMajor, Atrans, Btrans,
		p_rows, p_cols, p_common,
		1, p_A, lda,
		p_B, ldb,
		p_accumulate ? 1 : 0, p_C, p_cols);
}