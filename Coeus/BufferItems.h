#pragma once
#include "Tensor.h"

namespace Coeus
{
	struct __declspec(dllexport) DCQItem
	{
		Tensor s0;
		Tensor a;
		Tensor s1;
		float r;
		bool final;

		DCQItem(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, const float p_r, const bool p_final) {
			s0 = Tensor(*p_s0);
			a = Tensor(*p_a);
			s1 = Tensor(*p_s1);
			r = p_r;
			final = p_final;
		}
	};

	struct __declspec(dllexport) DQItem
	{
		Tensor s0;
		Tensor a;
		Tensor s1;
		float r;
		bool final;

		DQItem(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, const float p_r, const bool p_final) {
			s0 = Tensor(*p_s0);
			a = Tensor(*p_a);
			s1 = Tensor(*p_s1);
			r = p_r;
			final = p_final;
		}
	};

	struct __declspec(dllexport) TransitionItem
	{
		Tensor s0;
		Tensor a;
		Tensor s1;

		TransitionItem(Tensor* p_s0, Tensor* p_a, Tensor* p_s1) {
			s0 = Tensor(*p_s0);
			a = Tensor(*p_a);
			s1 = Tensor(*p_s1);
		}
	};
}

