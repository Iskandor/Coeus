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

BaseCellGroup::BaseCellGroup(int p_dim, bool p_bias): 
	_dim(p_dim), 
	_f(nullptr),
	_bias_flag(p_bias)
{
	_id = IDGen::instance().next();
	_net = Tensor::Zero({p_dim});
	_output = Tensor::Zero({p_dim});
	_deriv_input = Tensor::Zero({p_dim});

	if (p_bias)
	{
		_bias = add_param(_id, new Tensor({ _dim }, Tensor::RANDOM, 1));
	}
}

BaseCellGroup::BaseCellGroup(BaseCellGroup* p_source):
	_dim(p_source->_dim),
	_f(nullptr),
	_bias_flag(p_source->_bias_flag)
{
	_id = p_source->_id;
	_net = Tensor::Zero({ p_source->_dim });
	_output = Tensor::Zero({ p_source->_dim });
	_deriv_input = Tensor::Zero({ p_source->_dim });

	if (_bias_flag)
	{
		_bias = add_param(_id, p_source->_bias);
	}
}

BaseCellGroup::BaseCellGroup(nlohmann::json p_data): 
	_f(nullptr)
{
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_bias_flag = p_data["bias_flag"].get<bool>();

	if (_bias_flag)
	{
		double* data = Tensor::alloc_arr(_dim);

		stringstream ss(p_data["bias"].get<string>());

		ss.seekg(0, ios::end);
		const streampos size = ss.tellg();
		ss.seekg(0, ios::beg);
		ss.read((char*)(data), size);

		_bias = add_param(_id, new Tensor({ _dim }, data));
	}

	_output = Tensor::Zero({ _dim });
	_deriv_input = Tensor::Zero({ _dim });
	_net = Tensor::Zero({ _dim });
}

BaseCellGroup::~BaseCellGroup()
= default;

void BaseCellGroup::set_output(Tensor* p_output) const
{
	_output.override(p_output);
}

void BaseCellGroup::set_output(vector<Tensor*>& p_output) const
{
	int index = 0;

	for (auto& tensor : p_output)
	{
		for(int j = 0; j < tensor->size(); j++)
		{
			_output.set(index, tensor->at(j));
			index++;
		}		
	}
}

void BaseCellGroup::update_bias(Tensor& p_delta_b) const
{
	*_bias += p_delta_b;
}

json BaseCellGroup::get_json() const
{
	json data;

	data["id"] = _id;
	data["dim"] = _dim;
	data["bias_flag"] = _bias_flag;

	if (_bias_flag)
	{
		stringstream ss;

		for (int i = 0; i < _bias->size(); i++) {
			double b = (*_bias)[i];
			ss.write(reinterpret_cast<char*>(&b), sizeof(double));
		}

		data["bias"] = ss.str();
	}

	return data;
}

void BaseCellGroup::copy(const BaseCellGroup& p_copy)
{
	_id = p_copy._id;
	_dim = p_copy._dim;
	_bias_flag = p_copy._bias_flag;
	_bias->override(p_copy._bias);

	_output = Tensor::Zero({ _dim });
	_deriv_input = Tensor::Zero({ _dim });
	_net = Tensor::Zero({ _dim });
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
