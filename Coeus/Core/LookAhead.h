#pragma once
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) LookAhead : GradientAlgorithm
	{
	public:
		LookAhead(NeuralNetwork* p_network);
		~LookAhead();

		float train(Tensor* p_input, Tensor* p_target) override;
		float train(vector<Tensor*>* p_input, Tensor* p_target) override;
		float train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, bool p_update) override;

		void init(GRADIENT_RULE p_update_rule, ICostFunction* p_cost_function, float p_alpha, int p_k = 6);

	private:
		void slow_update();
		int _k;
		int _kt;

		float _error;

		map<string, Tensor> _slow_params;
	};
}
