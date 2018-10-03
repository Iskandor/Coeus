#pragma once
#include "Tensor.h"
#include <string>
#include "Coeus.h"
#include "IActivationFunction.h"
#include "json.hpp"

using namespace FLAB;

namespace Coeus {
	class __declspec(dllexport) BaseCellGroup
	{
	public:
		BaseCellGroup(int p_dim);
		explicit BaseCellGroup(nlohmann::json p_data);
		// 'Coeus::BaseCellGroup': cannot instantiate abstract class
		//BaseCellGroup(BaseCellGroup& p_copy) = delete;
		//BaseCellGroup& operator = (const BaseCellGroup& p_copy) = delete;
		template<typename T>
		static BaseCellGroup *clone(const BaseCellGroup *obj);

		virtual ~BaseCellGroup();


		virtual void integrate(Tensor* p_input, Tensor* p_weights) = 0;
		virtual void activate() = 0;

		string	get_id() const { return _id; }
		int		get_dim() const { return _dim; }

		void	set_output(Tensor* p_output) const;
		Tensor* get_output() { return &_output; }

	protected:
		void copy(const BaseCellGroup& p_copy);
		static IActivationFunction* init_activation_function(ACTIVATION p_activation_function);

		string  _id;
		int     _dim;

		Tensor	_net;
		Tensor	_output;

	};
}

