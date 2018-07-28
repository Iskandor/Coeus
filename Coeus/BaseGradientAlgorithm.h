#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "NetworkGradient.h"

namespace Coeus {

	class __declspec(dllexport) BaseGradientAlgorithm
	{
	public:
		BaseGradientAlgorithm(NeuralNetwork* p_network);
		virtual ~BaseGradientAlgorithm();

		double train(Tensor* p_input, Tensor* p_target);
		double train(vector<Tensor*>* p_input, Tensor* p_target);
		double train(vector<Tensor*>* p_input, vector<Tensor*>* p_target);

	protected:
		double train(Tensor* p_target);
		void init(ICostFunction* p_cost_function, const double p_alpha);
		virtual void calc_update();
		virtual void init_structures();

		NeuralNetwork*		_network;
		ICostFunction*		_cost_function;
		NetworkGradient*	_network_gradient;

		map<string, Tensor> _update;
		map<string, Tensor> _update_batch;

		double _alpha;

	private:
		bool _init_structures;
		int	 _batch;
		
	};
}

