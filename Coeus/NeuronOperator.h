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
		explicit NeuronOperator(json p_data);
		NeuronOperator(NeuronOperator& p_copy);
		~NeuronOperator();

		virtual void integrate(Tensor* p_input, Tensor* p_weights);
		virtual void activate();
		Tensor derivative() const;

		IActivationFunction* get_function() const { return _activation_function; }
		Tensor* get_output() const { return _output; }
		Param*	get_bias() const { return _bias; }
		string	get_id() const { return _id; }

		json get_json() const;

		static Tensor* init_auxiliary_parameter(Tensor* p_param, int p_rows, int p_cols);
		static Tensor* init_auxiliary_parameter(Tensor* p_param, int p_depth, int p_rows, int p_cols);
		static Tensor* init_auxiliary_parameter(Tensor* p_param, int p_batch, int p_depth, int p_rows, int p_cols);

	protected:
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

