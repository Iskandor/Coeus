#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) Adadelta : public BaseGradientAlgorithm
	{
	public:
		explicit Adadelta(NeuralNetwork* p_network);
		~Adadelta();

		void init(ICostFunction* p_cost_function, double p_alpha = 1, double p_decay = 0.9, double p_epsilon = 1e-8);

	private:
		void update_cache(string p_id, Tensor &p_gradient);
		void update_cache_delta(string p_id, Tensor &p_gradient);
		void calc_update() override;
		void init_structures() override;

		double _decay;
		double _epsilon;

		map<string, Tensor> _cache;
		map<string, Tensor> _cache_delta;

	};
}

