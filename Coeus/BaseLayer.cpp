#include "BaseLayer.h"
#include "NeuronOperator.h"
#include "TensorOperator.h"

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

		for (int dim : p_in_dim)
		{
			_in_dim_tensor->set(i, dim);
			i++;
		}
	}

	_input = nullptr;
	_output = nullptr;
	_is_recurrent = false;
	_mode = NONE;
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
	_mode = NONE;
}

BaseLayer::~BaseLayer()
{
	delete _dim_tensor;
	delete _input;

	for (const auto& it : _delta_in)
	{
		delete it.second;
	}

	for (const auto& it : _delta)
	{
		delete it.second;
	}

	delete _delta_out;
}

void BaseLayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
{
	if (!p_input_layers.empty())
	{
		_in_dim = 0;

		for (auto it : p_input_layers)
		{
			_in_dim += it->get_dim();
			_input_layer.push_back(it);
			_delta_in[it->get_id()] = nullptr;
		}
	}

	_delta_out = nullptr;
	if (!p_output_layers.empty())
	{
		for (auto it : p_output_layers)
		{
			_output_layer.push_back(it);
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

void BaseLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
	if (_output_layer.size() == 1)
	{
		_delta_out = NeuronOperator::init_auxiliary_parameter(_delta_out, _batch_size, _dim);
		_delta_out->override(_output_layer[0]->get_delta_in(_id));
	}
	if (_output_layer.size() > 1)
	{
		_delta_out = NeuronOperator::init_auxiliary_parameter(_delta_out, _batch_size, _dim);
		_delta_out->fill(0);
		for (BaseLayer* it : _output_layer)
		{
			if (it->get_delta_in(_id) != nullptr)
			{
				TensorOperator::instance().vv_add(_delta_out->arr(), it->get_delta_in(_id)->arr(), _delta_out->arr(), _delta_out->size());
			}			
		}
	}
}

Tensor* BaseLayer::get_delta_in(const string& p_id)
{
	return _delta_in[p_id];
}

void BaseLayer::set_delta_out(Tensor* p_value)
{
	if (p_value != nullptr)
	{
		if (p_value->rank() == 1)
		{
			_delta_out = NeuronOperator::init_auxiliary_parameter(_delta_out, 1, _dim);
		}
		if (p_value->rank() == 2)
		{
			_delta_out = NeuronOperator::init_auxiliary_parameter(_delta_out, p_value->shape(0), _dim);
		}
		_delta_out->override(p_value);
	}
	else
	{
		if (_delta_out == nullptr)
		{
			_delta_out = new Tensor(_output->rank(), Tensor::copy_shape(_output->rank(), _output->shape()), Tensor::ONES);
		}
		
	}
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

vector<string> BaseLayer::unfold_layer()
{
	vector<string> result;

	for(BaseLayer* layer : _input_layer)
	{
		result.push_back(layer->get_id());
	}

	return vector<string>(result);
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
