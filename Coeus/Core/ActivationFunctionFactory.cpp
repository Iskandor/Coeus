#include "ActivationFunctionFactory.h"
#include "SigmoidActivation.h"
#include "TanhActivation.h"
#include "LinearActivation.h"
#include "BinaryActivation.h"
#include "SoftmaxActivation.h"
#include "SoftplusActivation.h"
#include "ReluActivation.h"

using namespace Coeus;

IActivationFunction* ActivationFunctionFactory::create_function(const ACTIVATION p_type)
{
	IActivationFunction* result = nullptr;

	switch(p_type)
	{
	case LINEAR:
		result = new LinearActivation();
		break;
	case BINARY:
		result = new BinaryActivation();
		break;
	case SIGMOID:
		result = new SigmoidActivation();
		break;
	case TANH:
		result = new TanhActivation();
		break;
	case SOFTMAX:
		result = new SoftmaxActivation();
		break;
	case SOFTPLUS:
		result = new SoftplusActivation();
		break;
	case RELU:
		result = new ReluActivation();
		break;
	case EXP: break;
	case GAUSS: break;
	default: ;
	}

	return result;
}

ActivationFunctionFactory::ActivationFunctionFactory()
= default;


ActivationFunctionFactory::~ActivationFunctionFactory()
= default;
