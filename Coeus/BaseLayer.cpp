#include "BaseLayer.h"
#include "NeuronOperator.h"

using namespace Coeus;

BaseLayer::BaseLayer(const string& p_id, const int p_dim, const initializer_list<int> p_in_dim)
{
	_id = p_id;
	_dim = p_dim;
	_in_dim = sum_input_dim(p_in_dim);
	_input_dim = sum_input_dim(p_in_dim);
	_valid = false;
	_batch_size = 0;

	_dim_tensor = nullptr;
	_in_dim_tensor = nullptr;

	if (p_in_dim.size() != 0)
	{
		_in_dim_tensor = new Tensor({ static_cast<int>(p_in_dim.size()) }, Tensor::ZERO);

		int i = 0;

		for(auto it = p_in_dim.begin(); it != p_in_dim.end(); it++)
		{
			_in_dim_tensor->set(i, *it);
			i++;
		}
	}

	_input = nullptr;
	_output = nullptr;
	_is_recurrent = false;
	_learning_mode = false;
}

BaseLayer::BaseLayer(json p_data)
{
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_input_dim = p_data["in_dim"].get<int>();
	_in_dim = _input_dim;
	_valid = false;
	_batch_size = 0;

	_dim_tensor = nullptr;
	_in_dim_tensor = nullptr;

	_input = nullptr;
	_output = nullptr;
	_is_recurrent = false;
	_learning_mode = false;
}

BaseLayer::~BaseLayer()
{
	delete _dim_tensor;
	delete _input;
}

void BaseLayer::init(vector<BaseLayer*>& p_input_layers)
{
	if (!p_input_layers.empty())
	{
		_in_dim = 0;

		for (auto it : p_input_layers)
		{
			_in_dim += it->get_dim();
			_input_layer.push_back(it);
		}
	}
}

void BaseLayer::integrate(Tensor* p_input)
{
	if (p_input->rank() == 1)
	{
		_batch_size = 1;
		_batch = false;
	}
	if (p_input->rank() == 2)
	{
		_batch_size = p_input->shape(0);
		_batch = true;
	}
	if (p_input->rank() == 3)
	{
		_batch_size = 1;
		_batch = false;
	}

	_input = NeuronOperator::init_auxiliary_parameter(_input, _batch_size, _in_dim);

	_input->push_back(p_input);
}

json BaseLayer::get_json() const
{
	json data;

	data["id"] = _id;
	data["type"] = _type;
	data["dim"] = _dim;
	data["in_dim"] = _input_dim;

	return data;
}

BaseLayer::BaseLayer(BaseLayer* p_source)                                     
{
	_id = p_source->_id;
	_dim = p_source->_dim;
	_type = p_source->_type;
	_in_dim = p_source->_in_dim;
	_input_dim = p_source->_input_dim;
	_batch_size = p_source->_batch_size;
	_batch = p_source->_batch;
	_valid = false;

	_input = nullptr;
	_output = nullptr;
}

int BaseLayer::sum_input_dim(initializer_list<int> p_in_dim) const
{
	int result = 0;

	for(auto it = p_in_dim.begin(); it != p_in_dim.end(); it++)
	{
		result += *it;
	}

	return result;
}
