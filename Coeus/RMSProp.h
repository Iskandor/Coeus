#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) RMSProp: public BaseGradientAlgorithm
	{
	public:
		explicit RMSProp(NeuralNetwork* p_network);
		~RMSProp();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_decay = 0.9, double p_epsilon = 1e-8);

	private:
		void update_cache(string p_id, Tensor &p_gradient);
		void calc_update() override;

		double _decay;
		double _epsilon;

		map<string, Tensor> _cache;

	};
}

