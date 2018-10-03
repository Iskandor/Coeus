#include "BaseCellGroup.h"
#include "LinearActivation.h"
#include "BinaryActivation.h"
#include "SigmoidActivation.h"
#include "TanhActivation.h"
#include "SoftplusActivation.h"
#include "ReluActivation.h"
#include "SoftmaxActivation.h"
#include "IDGen.h"

using namespace Coeus;

BaseCellGroup::BaseCellGroup(int p_dim): _dim(p_dim)
{
	_id = IDGen::instance().next();
	_net = Tensor::Zero({p_dim});
	_output = Tensor::Zero({p_dim});
}

BaseCellGroup::BaseCellGroup(nlohmann::json p_data)
{
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_output = Tensor::Zero({ _dim });
	_net = Tensor::Zero({ _dim });
}

BaseCellGroup::~BaseCellGroup()
= default;

void BaseCellGroup::set_output(Tensor* p_output) const
{
	_output.override(p_output);
}

void BaseCellGroup::copy(const BaseCellGroup& p_copy)
{
	_id = p_copy._id;
	_dim = p_copy._dim;
}

template <typename T>
BaseCellGroup* BaseCellGroup::clone(const BaseCellGroup* obj)
{
	return new T(dynamic_cast<const T &>(*obj));
}

IActivationFunction* BaseCellGroup::init_activation_function(ACTIVATION p_activation_function)
{
	IActivationFunction* f;
	switch (p_activation_function) {
	case LINEAR:
		f = new LinearActivation();
		break;
	case BINARY:
		f = new BinaryActivation();
		break;
	case SIGMOID:
		f = new SigmoidActivation();
		break;
	case TANH:
		f = new TanhActivation();
		break;
	case SOFTPLUS:
		f = new SoftplusActivation();
		break;
	case RELU:
		f = new ReluActivation();
		break;
	case SOFTMAX:
		f = new SoftmaxActivation();
		break;
	default:
		f = nullptr;
	}

	return f;
}
