#include "PoolingLayer.h"
#include "NeuronOperator.h"

using namespace Coeus;

PoolingLayer::PoolingLayer(const string& p_id, const int p_extent, const int p_stride, const initializer_list<int> p_in_dim) : BaseLayer(p_id, p_extent, p_in_dim)
{
	_extent = p_extent;
	_stride = p_stride;
}

PoolingLayer::PoolingLayer(PoolingLayer& p_copy, bool p_clone) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._in_dim })
{
	_extent = p_copy._extent;
	_stride = p_copy._stride;
}

PoolingLayer::~PoolingLayer()
= default;

PoolingLayer* PoolingLayer::copy(const bool p_clone)
{
	return new PoolingLayer(*this, p_clone);
}

void PoolingLayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
{
	int d1 = 0;
	int h1 = 0;
	int w1 = 0;

	if (!p_input_layers.empty())
	{
		for (auto it : p_input_layers)
		{
			if (it->get_dim_tensor()->size() != 3)
			{
				assert(("Invalid input shape", 0));
			}
			else
			{
				d1 += it->get_dim_tensor()->at(0);
				h1 = it->get_dim_tensor()->at(1);
				w1 = it->get_dim_tensor()->at(2);
			}

			_input_layer.push_back(it);
		}

		_in_dim_tensor = new Tensor({ 3 }, Tensor::ZERO);
		_in_dim_tensor->set(0, d1);
		_in_dim_tensor->set(1, h1);
		_in_dim_tensor->set(2, w1);
	}
	else
	{
		if (get_in_dim_tensor() != nullptr)
		{
			if (get_in_dim_tensor()->size() != 3)
			{
				assert(("Invalid input shape", 0));
			}
			else
			{
				d1 = get_in_dim_tensor()->at(0);
				h1 = get_in_dim_tensor()->at(1);
				w1 = get_in_dim_tensor()->at(2);
			}
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

	int d2 = d1;
	int h2 = (h1 - _extent) / _stride + 1;
	int w2 = (w1 - _extent) / _stride + 1;

	_dim_tensor = new Tensor({ 3 }, Tensor::ZERO);
	_dim_tensor->set(0, d2);
	_dim_tensor->set(1, h2);
	_dim_tensor->set(2, w2);

	_dim = d2 * h2 * w2;

	cout << _id << " " << *_in_dim_tensor << " - " << *_dim_tensor << endl;
}

void PoolingLayer::integrate(Tensor* p_input)
{
	if (p_input->rank() == 3)
	{
		_batch_size = 1;
		_input = NeuronOperator::init_auxiliary_parameter(_input, _batch_size, p_input->shape(0), p_input->shape(1), p_input->shape(2));
		_batch = false;
	}
	if (p_input->rank() == 4)
	{
		_batch_size = p_input->shape(0);
		_input = NeuronOperator::init_auxiliary_parameter(_input, _batch_size, p_input->shape(0), p_input->shape(1), p_input->shape(2));
		_batch = true;
	}

	_input->push_back(p_input);
}

void PoolingLayer::activate()
{
	_input->reset_index();
	_max_index.clear();

	int d1 = _input->shape(0);
	int h1 = _input->shape(1);
	int w1 = _input->shape(2);

	int d2 = d1;
	int h2 = (h1 - _extent) / _stride + 1;
	int w2 = (w1 - _extent) / _stride + 1;

	_output = NeuronOperator::init_auxiliary_parameter(_output, d2, h2, w2);

	for(int d = 0; d < d2; d++)
	{
		Tensor input_slice = _input->slice(d);

		for (int h = 0; h < h2; h++)
		{
			for (int w = 0; w < w2; w++)
			{
				int index = d * h1 * w1 + Tensor::subregion_max_index(&input_slice, h * _stride, w * _stride, _extent, _extent);
				_max_index.push_back(index);
				_output->set(d, h, w, _input->at(index));
			}
		}
	}
}

void PoolingLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
}

void PoolingLayer::calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
	BaseLayer::calc_gradient(p_gradient_map, p_derivative_map);

	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, d1, h1, w1);
		delta_in->fill(0);

		for(int i = 0; i < _delta_out->size(); i++)
		{
			delta_in->set(_max_index[i], _delta_out->at(i));
		}

		int index = 0;

		for (auto it : _input_layer)
		{
			_delta_in[it->get_id()] = NeuronOperator::init_auxiliary_parameter(_delta_in[it->get_id()], _batch_size, it->get_dim());
			delta_in->splice(index, _delta_in[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
	}
}

void PoolingLayer::reset()
{
}

void PoolingLayer::copy_params(BaseLayer* p_source)
{
}

Tensor* PoolingLayer::get_dim_tensor()
{
	return _dim_tensor;
}

json PoolingLayer::get_json() const
{
	json data = BaseLayer::get_json();

	return data;
}
