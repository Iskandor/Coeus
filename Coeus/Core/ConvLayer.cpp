#include "ConvLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"
#include "TensorOperator.h"
#include <chrono>

using namespace Coeus;

ConvLayer::ConvLayer(const string& p_id, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_filters, const int p_extent, const int p_stride, const int p_padding, const initializer_list<int> p_in_dim) : BaseLayer(p_id, p_filters, p_in_dim)
{
	_type = CONV;
	_filters = p_filters;
	_extent = p_extent;
	_stride = p_stride;
	_padding = p_padding;

	_initializer = p_initializer;
	_filter_input = nullptr;
	_column_input = nullptr;
	_padded_input = nullptr;

	_y = new ConvOperator(_filters, p_activation);
	add_param(_y);
	_W = nullptr;
}

ConvLayer::~ConvLayer()
{
	delete _y;
	delete _initializer;

	delete _filter_input;
	delete _column_input;
	delete _padded_input;
}

ConvLayer* ConvLayer::clone()
{
	return nullptr;
}

void ConvLayer::init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers)
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

	_W = new Param(IDGen::instance().next(), new Tensor({d1 * _extent * _extent, _filters }, Tensor::ZERO));
	_initializer->init(_W->get_data());
	add_param(_W->get_id(), _W->get_data());

	int d2 = _filters;
	int h2 = (h1 - _extent + 2 * _padding) / _stride + 1;
	int w2 = (w1 - _extent + 2 * _padding) / _stride + 1;

	_dim_tensor = new Tensor({ 3 }, Tensor::ZERO);
	_dim_tensor->set(0, d2);
	_dim_tensor->set(1, h2);
	_dim_tensor->set(2, w2);

	_dim = d2 * h2 * w2;

	_column_input = new Tensor({ d1 * _extent * _extent, h2 * w2 }, Tensor::ZERO);
	_padded_input = new Tensor({ d1, h1 + 2 * _padding, w1 + 2 * _padding }, Tensor::ZERO);

	cout << _id << " " << *_in_dim_tensor << " - " << *_dim_tensor << endl;
}

void ConvLayer::integrate(Tensor* p_input)
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

void ConvLayer::activate()
{
	_input->reset_index();

	int d1 = _input->shape(0);
	int h1 = _input->shape(1);
	int w1 = _input->shape(2);	

	int d2 = _filters;
	int h2 = (h1 - _extent + 2 * _padding) / _stride + 1;
	int w2 = (w1 - _extent + 2 * _padding) / _stride + 1;
	
	_output = NeuronOperator::init_auxiliary_parameter(_output, d2, h2, w2);
	_output->fill(0);
	_filter_input = NeuronOperator::init_auxiliary_parameter(_filter_input, 1, _extent * _extent);

	auto start = chrono::high_resolution_clock::now();

	_padded_input->reset_index();

	for (int d = 0; d < d1; d++)
	{
		Tensor input_slice = _input->slice(d);
		if (_padding > 0)
		{
			input_slice.padding(_padding);
		}
		_padded_input->push_back(&input_slice);
	}

	im2col(_padded_input, _column_input);
	_y->integrate(_dim_tensor, d1 * _extent * _extent, h2 * w2, _column_input, _W->get_data());
	_y->activate();
	_output = _y->get_output();
}

void ConvLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
}

void ConvLayer::calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map)
{
	BaseLayer::calc_gradient(p_gradient_map, p_derivative_map);
	Tensor*	 df = _y->get_function()->backward(_delta_out);

	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	int d2 = _filters;
	int h2 = (_in_dim_tensor->at(1) - _extent + 2 * _padding) / _stride + 1;
	int w2 = (_in_dim_tensor->at(2) - _extent + 2 * _padding) / _stride + 1;

	Tensor gradient({ w2 * h2, _extent * _extent }, Tensor::ZERO);
	Tensor gradient_slice({ _extent * _extent }, Tensor::ZERO);

	TensorOperator::instance().M_reduce(p_gradient_map[_y->get_bias()->get_id()].arr(), df->arr(), true, d2, h2 * w2, false);

	TensorOperator::instance().MM_prod(df->arr(), false, _column_input->arr(), true, p_gradient_map[_W->get_id()].arr(), _filters, h2 * w2, d1 * _extent * _extent);

	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		TensorOperator::instance().MM_prod(_W->get_data()->arr(), false, df->arr(), false, _column_input->arr(), d1 * _extent * _extent, _filters, h2 * w2);

		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, d1, h1, w1);
		delta_in->reset_index();

		col2im(_column_input, delta_in);

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

void ConvLayer::override(BaseLayer* p_source)
{
}

void ConvLayer::reset()
{
}

json ConvLayer::get_json() const
{
	json data = BaseLayer::get_json();

	return data;
}

Tensor* ConvLayer::get_dim_tensor()
{
	return _dim_tensor;
}

void ConvLayer::im2col(Tensor* p_image, Tensor* column) const
{
	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	int d2 = _filters;
	int h2 = (_in_dim_tensor->at(1) - _extent + 2 * _padding) / _stride + 1;
	int w2 = (_in_dim_tensor->at(2) - _extent + 2 * _padding) / _stride + 1;
	
	
	Tensor region({ _extent * _extent }, Tensor::ZERO);
	column->reset_index();

	for (int h = 0; h < h2; h++)
	{
		for (int w = 0; w < w2; w++)
		{
			for (int d = 0; d < d1; d++)
			{
				Tensor::subregion(&region, p_image, d, h * _stride, w * _stride, _extent, _extent);
				column->push_back(&region);
			}
		}
	}
}

void ConvLayer::col2im(Tensor* p_column, Tensor* p_image) const
{
	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	int d2 = _filters;
	int h2 = (_in_dim_tensor->at(1) - _extent + 2 * _padding) / _stride + 1;
	int w2 = (_in_dim_tensor->at(2) - _extent + 2 * _padding) / _stride + 1;

	Tensor subregion({ d1, (h1 + 2 * _padding), (w1 + 2 * _padding) }, Tensor::ZERO);
	Tensor subregion_padding({ h1, w1 }, Tensor::ZERO);

	for (int h = 0; h < h2; h++)
	{
		for (int w = 0; w < w2; w++)
		{
			for (int d = 0; d < d1; d++)
			{
				Tensor::add_subregion(&subregion, d, h * _stride, w * _stride, _extent, _extent, p_column, d * _extent * _extent, h * w2 + w, _extent * _extent, 1);
			}
		}
	}

	p_image->reset_index();

	for (int d = 0; d < d1; d++)
	{
		Tensor slice = subregion.slice(d);
		Tensor::subregion(&subregion_padding, &slice, _padding, _padding, w1, h1);
		p_image->push_back(&subregion_padding);
	}
}
