#pragma once
#include "Tensor.h"

namespace Coeus
{
	class TensorFactory
	{
	public:
		static Tensor* tensor(int p_rows, Tensor* p_tensor = nullptr)
		{
			Tensor* result = p_tensor;

			if (result == nullptr || result->rank() != 1 || result->size() != p_rows)
			{
				delete result;
				result = new Tensor({ p_rows }, Tensor::ZERO);
			}		

			return result;
		}

		static Tensor* tensor(int p_rows, int p_cols, Tensor* p_tensor = nullptr)
		{
			Tensor* result = p_tensor;

			if (result == nullptr || result->rank() != 2 || result->size() != p_rows * p_cols)
			{
				delete result;
				result = new Tensor({ p_rows, p_cols }, Tensor::ZERO);
			}

			return result;
		}

		static Tensor* tensor(int p_depth, int p_rows, int p_cols, Tensor* p_tensor = nullptr)
		{
			Tensor* result = p_tensor;

			if (result == nullptr || result->rank() != 3 || result->size() != p_depth * p_rows * p_cols)
			{
				delete result;
				result = new Tensor({ p_depth, p_rows, p_cols }, Tensor::ZERO);
			}

			return result;
		}

		static Tensor* tensor(int p_batch, int p_depth, int p_rows, int p_cols, Tensor* p_tensor = nullptr)
		{
			Tensor* result = p_tensor;

			if (result == nullptr || result->rank() != 4 || result->size() != p_batch * p_depth * p_rows * p_cols)
			{
				delete result;
				result = new Tensor({ p_batch, p_depth, p_rows, p_cols }, Tensor::ZERO);
			}

			return result;
		}

	private:
		TensorFactory() = default;
		~TensorFactory() = default;
	};
}

