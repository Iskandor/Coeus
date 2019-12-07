#pragma once
#include "Tensor.h"

namespace Coeus
{
	class __declspec(dllexport) TensorInitializer
	{
	public:
		enum INIT {
			DEBUG = 0,
			UNIFORM = 1,
			LECUN_UNIFORM = 2,
			GLOROT_UNIFORM = 3,
			IDENTITY = 4,
			NORMAL = 5,
			EXPONENTIAL = 6,
			HE_UNIFORM = 7,
			LECUN_NORMAL = 8,
			GLOROT_NORMAL = 9,
			HE_NORMAL = 10
		};
		
		TensorInitializer(); 
		TensorInitializer(TensorInitializer &p_copy);
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
