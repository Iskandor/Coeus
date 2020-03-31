#include "tensor_operator_cpu.h"
#include <algorithm>
#include <mkl_cblas.h>
#include <cassert>
#include "CLAB.h"


void tensor_operator_cpu::add(float* p_x, const int p_x_size, float* p_y, const int p_y_size, float* p_z)
{
	if (p_x_size == p_y_size)
	{
		const int size = p_x_size / segment;

		if (p_x == p_z)
		{
			for (int i = 0; i < size; i++)
			{
				__m256 zx = _mm256_load_ps(p_z);
				const __m256 yx = _mm256_load_ps(p_y);

				zx = _mm256_add_ps(zx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_y += segment;
			}

			for (int i = size * segment; i < p_x_size; i++)
			{
				*p_z++ += *p_y++;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(p_x);
				const __m256 yx = _mm256_load_ps(p_y);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_add_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_x += segment;
				p_y += segment;
			}

			for (int i = size * segment; i < p_x_size; i++)
			{
				*p_z++ = *p_x++ + *p_y++;
			}
		}
	}
	if (p_x_size < p_y_size)
	{
		add_broadcast_x(p_x, p_x_size, p_y, p_y_size, p_z);
	}
	if (p_x_size > p_y_size)
	{
		add_broadcast_y(p_x, p_x_size, p_y, p_y_size, p_z);
	}
}

void tensor_operator_cpu::const_add(float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int size = p_size / segment * segment;
	const __m256 y = _mm256_broadcast_ss(&p_y);

	if (p_x == p_z)
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_add_ps(zx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ += p_y;
		}
	}
	else
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			const __m256 xx = _mm256_load_ps(p_x);
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_add_ps(xx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
			p_x += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ = *p_x++ + p_y;
		}
	}
}

void tensor_operator_cpu::sub(float* p_x, const int p_x_size, float* p_y, const int p_y_size, float* p_z)
{
	if (p_x_size == p_y_size)
	{
		const int size = p_x_size / segment;

		if (p_x == p_z)
		{
			for (int i = 0; i < size; i++)
			{
				__m256 zx = _mm256_load_ps(p_z);
				const __m256 yx = _mm256_load_ps(p_y);

				zx = _mm256_sub_ps(zx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_y += segment;
			}

			for (int i = size * segment; i < p_x_size; i++)
			{
				*p_z++ += *p_y++;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(p_x);
				const __m256 yx = _mm256_load_ps(p_y);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_sub_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_x += segment;
				p_y += segment;
			}

			for (int i = size * segment; i < p_x_size; i++)
			{
				*p_z++ = *p_x++ - *p_y++;
			}
		}
	}
	if (p_x_size < p_y_size)
	{
		sub_broadcast_x(p_x, p_x_size, p_y, p_y_size, p_z);
	}
	if (p_x_size > p_y_size)
	{
		sub_broadcast_y(p_x, p_x_size, p_y, p_y_size, p_z);
	}
}

void tensor_operator_cpu::const_sub(float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int size = p_size / segment * segment;
	const __m256 y = _mm256_broadcast_ss(&p_y);

	if (p_x == p_z)
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_sub_ps(zx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ -= p_y;
		}
	}
	else
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			const __m256 xx = _mm256_load_ps(p_x);
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_sub_ps(xx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
			p_x += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ = *p_x++ - p_y;
		}
	}
}

void tensor_operator_cpu::const_sub(float p_x, float* p_y, float* p_z, const int p_size)
{
	const int size = p_size / segment * segment;
	const __m256 x = _mm256_broadcast_ss(&p_x);

	if (p_y == p_z)
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			__m256 zy = _mm256_load_ps(p_y);

			zy = _mm256_sub_ps(x, zy);

			_mm256_storeu_ps(p_z, zy);

			p_z += segment;
			p_y += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z = p_x - *p_y;
			p_z++;
		}
	}
	else
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			const __m256 yx = _mm256_load_ps(p_y);
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_sub_ps(x, yx);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
			p_y += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ = p_x - *p_y++;
		}
	}
}

void tensor_operator_cpu::mul(float* p_x, bool p_transpose_x, float* p_y, bool p_transpose_y, float* p_z, int p_rows, int p_common, int p_cols)
{
	if (!p_transpose_x && !p_transpose_y)
	{
		mul_ab(p_x, p_y, p_z, p_rows, p_common, p_cols);
	}
	if (p_transpose_x && !p_transpose_y)
	{
		mul_aTb(p_x, p_y, p_z, p_rows, p_common, p_cols);
	}
	if (!p_transpose_x && p_transpose_y)
	{
		mul_abT(p_x, p_y, p_z, p_rows, p_common, p_cols);
	}

}

void tensor_operator_cpu::const_mul(float* p_x, const float p_y, float* p_z, const int p_size)
{
	const int size = p_size / segment * segment;
	const __m256 y = _mm256_broadcast_ss(&p_y);

	if (p_x == p_z)
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_mul_ps(zx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ *= p_y;
		}
	}
	else
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			const __m256 xx = _mm256_load_ps(p_x);
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_mul_ps(xx, y);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
			p_x += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ = *p_x++ * p_y;
		}
	}
}

void tensor_operator_cpu::const_div(float p_x, float* p_y, float* p_z, const int p_size)
{
	const int size = p_size / segment * segment;
	const __m256 x = _mm256_broadcast_ss(&p_x);

	if (p_y == p_z)
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			__m256 zy = _mm256_load_ps(p_y);

			zy = _mm256_div_ps(x, zy);

			_mm256_storeu_ps(p_z, zy);

			p_z += segment;
			p_y += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z = p_x / *p_y;
			p_z++;
		}
	}
	else
	{
		for (int i = 0; i < size / segment; i += segment)
		{
			const __m256 yx = _mm256_load_ps(p_y);
			__m256 zx = _mm256_load_ps(p_z);

			zx = _mm256_div_ps(x, yx);

			_mm256_storeu_ps(p_z, zx);

			p_z += segment;
			p_y += segment;
		}

		for (int i = size; i < p_size; i++)
		{
			*p_z++ = p_x / *p_y++;
		}
	}
}

void tensor_operator_cpu::reduce_sum(float* p_x, const int p_x_shape, float* p_y, const int p_y_size)
{
	if (p_x_shape > 1)
	{
		float* py = p_y;
		for (int j = 0; j < p_y_size; j++)
		{
			*py++ = 0;
		}

		const int size_y = p_y_size / segment;
		py = p_y;

		if (size_y > 0)
		{
			for (int i = 0; i < p_x_shape; i++)
			{
				py = p_y;
				for (int j = 0; j < size_y; j++)
				{
					__m256 yx = _mm256_load_ps(py);
					const __m256 xx = _mm256_load_ps(p_x);

					yx = _mm256_add_ps(xx, yx);
					_mm256_storeu_ps(py, yx);
					p_x += segment;
					py += segment;
				}
				for (int j = size_y * segment; j < p_y_size; j++)
				{
					*py++ += *p_x++;
				}

			}
		}
		else
		{
			for (int i = 0; i < p_x_shape; i++)
			{
				py = p_y;
				for (int j = 0; j < p_y_size; j++)
				{
					*py++ += *p_x++;
				}
			}
		}
	}
	else
	{
		memcpy(p_y, p_x, sizeof(float) * p_y_size);
	}
}

void tensor_operator_cpu::mul_ab(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols)
{
	const CBLAS_TRANSPOSE Atrans = CblasNoTrans;
	const CBLAS_TRANSPOSE Btrans = CblasNoTrans;

	const int lda = p_common;
	const int ldb = p_cols;

	cblas_sgemm(CblasRowMajor, Atrans, Btrans,
		p_rows, p_cols, p_common,
		1, p_x, lda,
		p_y, ldb,
		0, p_z, p_cols);
}

void tensor_operator_cpu::mul_aTb(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols)
{
	const CBLAS_TRANSPOSE Atrans = CblasTrans;
	const CBLAS_TRANSPOSE Btrans = CblasNoTrans;

	const int lda = p_rows;
	const int ldb = p_cols;

	cblas_sgemm(CblasRowMajor, Atrans, Btrans,
		p_rows, p_cols, p_common,
		1, p_x, lda,
		p_y, ldb,
		0, p_z, p_cols);
}

void tensor_operator_cpu::mul_abT(float* p_x, float* p_y, float* p_z, int p_rows, int p_common, int p_cols)
{
	const CBLAS_TRANSPOSE Atrans = CblasNoTrans;
	const CBLAS_TRANSPOSE Btrans = CblasTrans;

	const int lda = p_common;
	const int ldb = p_common;

	cblas_sgemm(CblasRowMajor, Atrans, Btrans,
		p_rows, p_cols, p_common,
		1, p_x, lda,
		p_y, ldb,
		0, p_z, p_cols);
}

void tensor_operator_cpu::add_broadcast_x(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z)
{
	const int size = p_x_size >= segment ? p_y_size / p_x_size : 0;

	float *px = p_x;

	if (p_x == p_z)
	{
		assert(("Unable to broadcast x for add(x,y,x) operation", 0));
	}
	else
	{
		if (size > 0)
		{
			int x_index = 0;
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(px);
				const __m256 yx = _mm256_load_ps(p_y);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_add_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				px += segment;
				p_y += segment;
				x_index += segment;

				if (x_index == p_x_size)
				{
					x_index = 0;
					px = p_x;
				}
				else if (p_x_size - x_index < segment)
				{
					for (int j = 0; j < p_x_size - x_index; j++)
					{
						*p_z++ = *px++ + *p_y++;
					}
					x_index = 0;
					px = p_x;
				}
			}
		}
		else
		{
			int x_index = 0;
			for (int i = 0; i < p_y_size; i++)
			{
				*p_z++ = *px++ + *p_y++;
				x_index++;
				if (x_index == p_x_size)
				{
					x_index = 0;
					px = p_x;
				}
			}
		}
	}
}

void tensor_operator_cpu::add_broadcast_y(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z)
{
	const int size = p_y_size >= segment ? p_x_size / p_y_size : 0;

	float *py = p_y;

	if (p_x == p_z)
	{
		if (size > 0)
		{
			int y_index = 0;
			for (int i = 0; i < size; i++)
			{
				__m256 zx = _mm256_load_ps(p_z);
				const __m256 yx = _mm256_load_ps(py);

				zx = _mm256_add_ps(zx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				py += segment;
				y_index += segment;

				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
				else if (p_y_size - y_index < segment)
				{
					for (int j = 0; j < p_y_size - y_index; j++)
					{
						*p_z++ += *py++;
					}
					y_index = 0;
					py = p_y;
				}
			}
		}
		else
		{
			int y_index = 0;
			for (int i = 0; i < p_x_size; i++)
			{
				*p_z++ += *py++;
				y_index++;
				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
			}
		}
	}
	else
	{
		if (size > 0)
		{
			int y_index = 0;
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(p_x);
				const __m256 yx = _mm256_load_ps(py);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_add_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_x += segment;
				py += segment;
				y_index += segment;

				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
				else if (p_y_size - y_index < segment)
				{
					for (int j = 0; j < p_y_size - y_index; j++)
					{
						*p_z++ = *p_x++ + *py++;
					}
					y_index = 0;
					py = p_y;
				}
			}
		}
		else
		{
			int y_index = 0;
			for (int i = 0; i < p_x_size; i++)
			{
				*p_z++ = *p_x++ + *py++;
				y_index++;
				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
			}
		}
	}
}

void tensor_operator_cpu::sub_broadcast_x(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z)
{
	const int size = p_x_size >= segment ? p_y_size / p_x_size : 0;

	float *px = p_x;

	if (p_x == p_z)
	{
		assert(("Unable to broadcast x for sub(x,y,x) operation", 0));
	}
	else
	{
		if (size > 0)
		{
			int x_index = 0;
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(px);
				const __m256 yx = _mm256_load_ps(p_y);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_sub_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				px += segment;
				p_y += segment;
				x_index += segment;

				if (x_index == p_x_size)
				{
					x_index = 0;
					px = p_x;
				}
				else if (p_x_size - x_index < segment)
				{
					for (int j = 0; j < p_x_size - x_index; j++)
					{
						*p_z++ = *px++ - *p_y++;
					}
					x_index = 0;
					px = p_x;
				}
			}
		}
		else
		{
			int x_index = 0;
			for (int i = 0; i < p_y_size; i++)
			{
				*p_z++ = *px++ - *p_y++;
				x_index++;
				if (x_index == p_x_size)
				{
					x_index = 0;
					px = p_x;
				}
			}
		}
	}
}

void tensor_operator_cpu::sub_broadcast_y(float* p_x, int p_x_size, float* p_y, int p_y_size, float* p_z)
{
	const int size = p_y_size >= segment ? p_x_size / p_y_size : 0;

	float *py = p_y;

	if (p_x == p_z)
	{
		if (size > 0)
		{
			int y_index = 0;
			for (int i = 0; i < size; i++)
			{
				__m256 zx = _mm256_load_ps(p_z);
				const __m256 yx = _mm256_load_ps(py);

				zx = _mm256_sub_ps(zx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				py += segment;
				y_index += segment;

				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
				else if (p_y_size - y_index < segment)
				{
					for (int j = 0; j < p_y_size - y_index; j++)
					{
						*p_z++ -= *py++;
					}
					y_index = 0;
					py = p_y;
				}
			}
		}
		else
		{
			int y_index = 0;
			for (int i = 0; i < p_x_size; i++)
			{
				*p_z++ -= *py++;
				y_index++;
				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
			}
		}
	}
	else
	{
		if (size > 0)
		{
			int y_index = 0;
			for (int i = 0; i < size; i++)
			{
				const __m256 xx = _mm256_load_ps(p_x);
				const __m256 yx = _mm256_load_ps(py);
				__m256 zx = _mm256_load_ps(p_z);

				zx = _mm256_sub_ps(xx, yx);

				_mm256_storeu_ps(p_z, zx);

				p_z += segment;
				p_x += segment;
				py += segment;
				y_index += segment;

				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
				else if (p_y_size - y_index < segment)
				{
					for (int j = 0; j < p_y_size - y_index; j++)
					{
						*p_z++ = *p_x++ - *py++;
					}
					y_index = 0;
					py = p_y;
				}
			}
		}
		else
		{
			int y_index = 0;
			for (int i = 0; i < p_x_size; i++)
			{
				*p_z++ = *p_x++ - *py++;
				y_index++;
				if (y_index == p_y_size)
				{
					y_index = 0;
					py = p_y;
				}
			}
		}
	}
}

tensor_operator_cpu::tensor_operator_cpu()
= default;


tensor_operator_cpu::~tensor_operator_cpu()
= default;
