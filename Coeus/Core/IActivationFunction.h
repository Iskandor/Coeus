#pragma once
#include "Tensor.h"
#include "Coeus.h"
#include "json.hpp"
#include "IGate.h"

using namespace nlohmann;

namespace Coeus
{
	class __declspec(dllexport) IActivationFunction : public IGate
	{
	public:
		explicit IActivationFunction(ACTIVATION p_type);
		virtual ~IActivationFunction();

		Tensor* forward(Tensor* p_input) override;
		Tensor* backward(Tensor* p_input, Tensor* p_x = nullptr) override;
		virtual Tensor derivative(Tensor& p_input) = 0;
		virtual json get_json();
		ACTIVATION get_type() const { return _type; }

	protected:
		Tensor*	_input;
		Tensor*	_output;
		Tensor* _gradient;
		ACTIVATION _type;

	};
}