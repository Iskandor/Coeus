#pragma once
#include "Tensor.h"
#include "Coeus.h"
#include "json.hpp"

using namespace FLAB;
using namespace nlohmann;

namespace Coeus {
	class __declspec(dllexport) IActivationFunction
	{
	public:
		explicit IActivationFunction(ACTIVATION p_type);
		virtual ~IActivationFunction();

		virtual Tensor activate(Tensor& p_input) = 0;
		virtual Tensor deriv(Tensor& p_input) = 0;
		virtual json get_json();
		ACTIVATION get_type() const { return _type; }		

	protected:
		ACTIVATION _type;
	};
}

