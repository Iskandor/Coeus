#pragma once
#include "Tensor.h"
#include "Coeus.h"

namespace Coeus
{
	class __declspec(dllexport) TensorInitializer
	{
	public:
		TensorInitializer();
		explicit TensorInitializer(INIT p_init, float p_arg1 = 0, float p_arg2 = 0);
		~TensorInitializer();

		static void init(Tensor* p_tensor, INIT p_init, float p_arg1 = 0, float p_arg2 = 0);
		void init(Tensor* p_tensor) const;

	private:
		static void uniform(Tensor* p_tensor, float p_min, float p_max);
		static void normal(Tensor* p_tensor, float p_mean, float p_dev);
		static void exponential(Tensor* p_tensor, float p_lambda);

		INIT _init;
		float _arg1{};
		float _arg2{};
	};
}
