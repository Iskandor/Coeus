#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"

namespace Coeus {
	class __declspec(dllexport) ParallelModule
	{
	public:
		ParallelModule(NeuralNetwork* p_network, IUpdateRule* p_update_rule, ICostFunction* p_cost_function, int p_batch);
		~ParallelModule();

		double run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target);
		map<string, Tensor>* get_update() { return &_update_batch; }

	private:
		void calc_update();

		int _batch;
		double*	_error;

		ICostFunction*		_cost_function;
		NeuralNetwork*		_network;
		NeuralNetwork**		_clone_network;
		NetworkGradient**	_network_gradient;
		IUpdateRule*		_update_rule;
		IUpdateRule**		_clone_update_rule;
		map<string, Tensor> _update_batch;
	};
}
