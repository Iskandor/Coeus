#pragma once
#include "NetworkGradient.h"

namespace Coeus {
	class __declspec(dllexport) NaturalGradient : public NetworkGradient
	{
	public:
		NaturalGradient(NeuralNetwork* p_network);
		~NaturalGradient();

		void calc_gradient(Tensor* p_loss = nullptr) override;
		void calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss = nullptr) override;

	private:

		map<string, Tensor> _fim;
		map<string, Tensor> _inv_fim;
		map<string, Tensor> _cache;
		int					_fim_n;
		float				_epsilon;
	};
}


