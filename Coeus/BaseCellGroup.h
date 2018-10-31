#pragma once
#include "Tensor.h"
#include <string>
#include "Coeus.h"
#include "IActivationFunction.h"
#include "json.hpp"
#include "ParamModel.h"

using namespace FLAB;

namespace Coeus {
	class __declspec(dllexport) BaseCellGroup : public ParamModel
	{
	public:
		BaseCellGroup(int p_dim, bool p_bias);
		explicit BaseCellGroup(nlohmann::json p_data);
		// 'Coeus::BaseCellGroup': cannot instantiate abstract class
		//BaseCellGroup(BaseCellGroup& p_copy) = delete;
		//BaseCellGroup& operator = (const BaseCellGroup& p_copy) = delete;
		virtual ~BaseCellGroup();


		virtual void integrate(Tensor* p_input, Tensor* p_weights) = 0;
		virtual void activate() = 0;

		string	get_id() const { return _id; }
		int		get_dim() const { return _dim; }

		void	set_output(Tensor* p_output) const;
		void	set_output(vector<Tensor*>& p_output) const;
		Tensor* get_output() { return &_output; }
		Tensor  get_deriv_output() { return _f->deriv(_deriv_input); }

		bool is_bias() const { return _bias_flag; }
		void update_bias(Tensor& p_delta_b) const;
		void set_bias(Tensor* p_bias) const { _bias->override(p_bias); }
		Tensor* get_bias() { return _bias; }

		IActivationFunction* get_activation_function() const { return _f; }

		virtual json get_json() const;

	protected:
		explicit BaseCellGroup(BaseCellGroup* p_source);
		void copy(const BaseCellGroup& p_copy);
		static IActivationFunction* init_activation_function(ACTIVATION p_activation_function);		

		string  _id;
		int     _dim;

		Tensor					_net;
		IActivationFunction*	_f;
		Tensor					_output;
		Tensor					_deriv_input;
		bool					_bias_flag{};
		Tensor*					_bias;

	};
}

