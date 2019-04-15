#pragma once
#include "Param.h"
#include "IActivationFunction.h"
#include "ParamModel.h"

namespace Coeus
{
	class __declspec(dllexport) NeuronOperator : public ParamModel
	{
	public:
		NeuronOperator(int p_dim, ACTIVATION p_activation);
		NeuronOperator(NeuronOperator& p_copy);
		~NeuronOperator();

		void integrate(Tensor* p_input, Tensor* p_weights);
		void activate();
		Tensor derivative() const;

		Tensor* get_output() const { return _output; }
		Param*	get_bias() const { return _bias; }
		string	get_id() const { return _id; }

		static Tensor* init_auxiliary_parameter(Tensor* p_param, int p_rows, int p_cols);

	private:
		string	_id;
		int		_dim;
		Param*	_bias;
		IActivationFunction* _activation_function;

		Tensor*	_net;
		Tensor*	_dnet;
		Tensor*	_int;
		Tensor*	_output;

	};
}

