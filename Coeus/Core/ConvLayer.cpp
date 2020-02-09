#include "ConvLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"
#include "TensorOperator.h"
#include <chrono>
#include "TensorFactory.h"

using namespace Coeus;

ConvLayer::ConvLayer(const string& p_id, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_filters, const int p_extent, const int p_stride, const int p_padding, const initializer_list<int> p_in_dim) : BaseLayer(p_id, p_filters, p_in_dim)
{
	_type = CONV;
	_filters = p_filters;
	_extent = p_extent;
	_stride = p_stride;
	_padding = p_padding;

	_initializer = p_initializer;
	_column_input = nullptr;
	_padded_input = nullptr;

	_y = new ConvOperator(_filters, p_activation);
	add_param(_y);
	_W = nullptr;
}

ConvLayer::ConvLayer(ConvLayer& p_copy, const bool p_clone) : BaseLayer(p_copy._id, p_copy._dim, { p_copy._in_dim })
{
	_type = CONV;
	_filters = p_copy._filters;
	_extent = p_copy._extent;
	_stride = p_copy._stride;
	_padding = p_copy._padding;

	_initializer = new TensorInitializer(*p_copy._initializer);
	_column_input = nullptr;
	_padded_input = nullptr;

	_y = new ConvOperator(*p_copy._y, p_clone);
	add_param(_y);	

	if (p_clone)
	{
		_W = new Param(IDGen::instance().next(), new Tensor(*p_copy._W->get_data()));
		_param_map[_W->get_id()] = p_copy._W->get_id();
		_param_map[_y->get_bias()->get_id()] = p_copy._y->get_bias()->get_id();
	}
	else
	{
		_W = new Param(p_copy._W->get_id(), p_copy._W->get_data());
	}
	add_param(_W);
}

ConvLayer::~ConvLayer()
{
	delete _y;
	delete _W;
	delete _initializer;

	delete _column_input;
	delete _padded_input;
}

ConvLayer* ConvLayer::copy(const bool p_clone)
{
	return new ConvLayer(*this, p_clone);
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

	if (_W == nullptr)
	{
		_W = new Param(IDGen::instance().next(), new Tensor({ d1 * _extent * _extent, _filters }, Tensor::ZERO));
		_initializer->init(_W->get_data());
		add_param(_W->get_id(), _W->get_data());
	}

	int d2 = _filters;
	int h2 = (h1 - _extent + 2 * _padding) / _stride + 1;
	int w2 = (w1 - _extent + 2 * _padding) / _stride + 1;

	_dim_tensor = new Tensor({ 3 }, Tensor::ZERO);
	_dim_tensor->set(0, d2);
	_dim_tensor->set(1, h2);
	_dim_tensor->set(2, w2);

	_dim = d2 * h2 * w2;

	cout << _id << " " << *_in_dim_tensor << " - " << *_dim_tensor << endl;
}

void ConvLayer::integrate(Tensor* p_input)
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

void ConvLayer::activate()
{
	_input->reset_index();

	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);

	int d2 = _filters;
	int h2 = (h1 - _extent + 2 * _padding) / _stride + 1;
	int w2 = (w1 - _extent + 2 * _padding) / _stride + 1;

	_padded_input = TensorFactory::tensor(_batch_size, d1, h1 + 2 * _padding, w1 + 2 * _padding, _padded_input);
	_column_input = TensorFactory::tensor(_batch_size * h2 * w2, d1 * _extent * _extent, _column_input);

	_output = TensorFactory::tensor(_batch_size, d2, h2, w2, _output);
	_output->fill(0);

	Tensor::padding(*_padded_input, *_input, _padding);

	Tensor::im2col(_padded_input, _column_input, _extent, _padding, _stride);
	_y->integrate(_dim_tensor, d1 * _extent * _extent, _batch_size * h2 * w2, _column_input, _W->get_data());
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

	df->reshape({ _batch_size, h2, w2, d2 });
	
	//TensorOperator::instance().M_reduce(p_gradient_map[_y->get_bias()->get_id()].arr(), df->arr(), true, d2, _batch_size * h2 * w2, false);
	TensorOperator::instance().conv_b_gradient(_batch_size, df->arr(), p_gradient_map[_y->get_bias()->get_id()].arr(), d2, h2, w2);

	TensorOperator::instance().MM_prod(df->arr(), true, _column_input->arr(), false, p_gradient_map[_W->get_id()].arr(), d2, _batch_size * h2 * w2, d1 * _extent * _extent);

	df->reshape({ _batch_size * h2 * w2, d2 });

	//p_gradient_map[_W->get_id()] = df->T() * *_column_input;

	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		TensorOperator::instance().MM_prod(df->arr(), false, _W->get_data()->arr(), true, _column_input->arr(), _batch_size * h2 * w2, d2, d1 * _extent * _extent);

		delta_in = TensorFactory::tensor(_batch_size, d1, h1, w1, delta_in);
		delta_in->reset_index();

		Tensor::col2im(_column_input, delta_in, _extent, _padding, _stride);
		delta_in->reshape({ _batch_size,  d1 * h1 * w1 });


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