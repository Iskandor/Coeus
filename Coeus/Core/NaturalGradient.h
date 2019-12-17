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
		void calc_gradient(Gradient& p_gradient);
		Gradient& get_gradient() override;
		map<string, Tensor>& get_hessian_inv() { return _inv_fim; }

	private:

		map<string, Tensor> _fim;
		map<string, Tensor> _inv_fim;
		Gradient			_natural_gradient;
		float				_epsilon;
		float				_alpha;
	};
}


