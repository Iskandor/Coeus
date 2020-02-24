#include "PoolingLayer.h"
#include "NeuronOperator.h"
#include "TensorFactory.h"

using namespace Coeus;

PoolingLayer::PoolingLayer(const string& p_id, const int p_extent, const int p_stride, const initializer_list<int> p_in_dim) : BaseLayer(p_id, p_extent, p_in_dim)
{
	_extent = p_extent;
	_stride = p_stride;
}

PoolingLayer::PoolingLayer(PoolingLayer& p_copy, bool p_clone) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._input_dim })
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
		_input = TensorFactory::tensor(_batch_size, p_input->shape(0), p_input->shape(1), p_input->shape(2), _input);
		_batch = false;
	}
	if (p_input->rank() == 4)
	{
		_batch_size = p_input->shape(0);
		_input = TensorFactory::tensor(_batch_size, p_input->shape(1), p_input->shape(2), p_input->shape(3), _input);
		_batch = true;
	}

	_input->push_back(p_input);
}

void PoolingLayer::activate()
{
	_input->reset_index();
	_max_index.clear();

	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	int d2 = d1;
	int h2 = (h1 - _extent) / _stride + 1;
	int w2 = (w1 - _extent) / _stride + 1;

	_output = TensorFactory::tensor(_batch_size, d2, h2, w2, _output);

	for(int n = 0; n < _batch_size; n++)
	{
		for (int d = 0; d < d2; d++)
		{
			for (int h = 0; h < h2; h++)
			{
				for (int w = 0; w < w2; w++)
				{
					//int index = d * h1 * w1 + Tensor::subregion_max_index(&input_slice, h * _stride, w * _stride, _extent, _extent);
					const int index = n * d1 * h1 * w1 + d * h1 * w1 + h * _stride * w1 + w * _stride;
					int max = index;

					for(int i = 0; i < _extent; i++)
					{
						for(int j = 0; j < _extent; j++)
						{
							if (_input->arr()[index + i * w1 + j] > _input->arr()[max])
							{
								max = index + i * w1 + j;
							}
						}
					}
					
					_max_index.push_back(max);
					_output->set(n, d, h, w, _input->at(max));
				}
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

	if (!_input_layer.empty())
	{
		Tensor* delta_in = TensorFactory::tensor(_batch_size, d1 * h1 * w1);
		delta_in->fill(0);

		for(int i = 0; i < _batch_size * _delta_out->size(); i++)
		{
			delta_in->set(_max_index[i], _delta_out->at(i));
		}

		int index = 0;

		for (auto it : _input_layer)
		{
			_delta_in[it->get_id()] = TensorFactory::tensor(_batch_size, it->get_dim(), _delta_in[it->get_id()]);
			delta_in->splice(index, _delta_in[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
	}
}

void PoolingLayer::reset()
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
