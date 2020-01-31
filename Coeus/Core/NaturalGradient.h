/*
* Natural gradient algorithm
* https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
*/
#pragma once
#include "NetworkGradient.h"

namespace Coeus {
	class __declspec(dllexport) NaturalGradient : public NetworkGradient
	{
	public:
		NaturalGradient(NeuralNetwork* p_network);
		virtual ~NaturalGradient();

		void calc_gradient(Tensor* p_loss = nullptr) override;
		static void calc_hessian(Gradient& p_gradient);

	private:
		static int _n;
	};
}


