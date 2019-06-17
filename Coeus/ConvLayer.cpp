#include "ConvLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"
#include "TensorOperator.h"

using namespace Coeus;

ConvLayer::ConvLayer(const string& p_id, const ACTIVATION p_activation, TensorInitializer* p_initializer, const int p_filters, const int p_extent, const int p_stride, const int p_padding, const initializer_list<int> p_in_dim) : BaseLayer(p_id, p_filters, p_in_dim)
{
	_type = CONV;
	_filters = p_filters;
	_extent = p_extent;
	_stride = p_stride;
	_padding = p_padding;

	_activation_function = ActivationFunctionFactory::create_function(p_activation);
	_initializer = p_initializer;
	_filter_input = nullptr;

	_y = new NeuronOperator*[_filters];

	for (int i = 0; i < _filters; i++)
	{
		_y[i] = new NeuronOperator(1, LINEAR);
		add_param(_y[i]);
	}
}

ConvLayer::~ConvLayer()
{
	for (int i = 0; i < _filters; i++)
	{
		delete _y[i];
		delete _W[i];
	}

	delete _y;
	delete _W;
	delete _initializer;

	delete _filter_input;

	delete _activation_function;
}

ConvLayer* ConvLayer::clone()
{
	return nullptr;
}

void ConvLayer::init(vector<BaseLayer*>& p_input_layers)
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


	_W = new Param*[_filters * d1];

	for (int i = 0; i < _filters * d1; i++)
	{
		_W[i] = new Param(IDGen::instance().next(), new Tensor({ 1, _extent * _extent }, Tensor::ZERO));
		_initializer->init(_W[i]->get_data());
		add_param(_W[i]->get_id(), _W[i]->get_data());
	}

	int d2 = _filters;
	int h2 = (h1 - _extent + 2 * _padding) / _stride + 1;
	int w2 = (w1 - _extent + 2 * _padding) / _stride + 1;

	_dim_tensor = new Tensor({ 3 }, Tensor::ZERO);
	_dim_tensor->set(0, d2);
	_dim_tensor->set(1, h2);
	_dim_tensor->set(2, w2);

	_dim = d2 * h2 * w2;
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

	for(int d = 0; d < d1; d++)
	{
		Tensor input_slice = _input->slice(d);
		if (_padding > 0)
		{
			input_slice.padding(_padding);
		}
		
		for (int h = 0; h < h2; h++)
		{
			for (int w = 0; w < w2; w++)
			{
				Tensor::subregion(_filter_input, &input_slice, h * _stride, w * _stride, _extent, _extent);
				
				for (int f = 0; f < _filters; f++)
				{
					_y[f]->integrate(_filter_input, _W[f * d1 + d]->get_data());
					_y[f]->activate();
					_output->set(f, h, w, _output->at(f, h, w) + _y[f]->get_output()->at(0));
				}
			}
		}
	}

	_output = _activation_function->forward(_output);
}

void ConvLayer::calc_derivative(map<string, Tensor*>& p_derivative)
{
}

void ConvLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	Tensor*	 delta_out = _activation_function->backward(p_delta_map[_id]);

	int d1 = _in_dim_tensor->at(0);
	int h1 = _in_dim_tensor->at(1);
	int w1 = _in_dim_tensor->at(2);


	int d2 = _filters;
	int h2 = (_in_dim_tensor->at(1) - _extent + 2 * _padding) / _stride + 1;
	int w2 = (_in_dim_tensor->at(2) - _extent + 2 * _padding) / _stride + 1;

	Tensor gradient({ w2 * h2, _extent * _extent }, Tensor::ZERO);
	Tensor gradient_slice({ _extent * _extent }, Tensor::ZERO);

	float err[1];

	for (int d = 0; d < d2; d++)
	{
		Tensor filter_error = delta_out->slice(d);
		TensorOperator::instance().v_reduce(p_gradient_map[_y[d]->get_bias()->get_id()].arr(), filter_error.arr(), filter_error.size());
	}


	for(int id = 0; id < d1; id++)
	{
		Tensor input_slice = _input->slice(id);
		if (_padding > 0)
		{
			input_slice.padding(_padding);
		}

		for (int d = 0; d < d2; d++)
		{
			Tensor filter_error = delta_out->slice(d);

			for (int h = 0; h < h2; h++)
			{
				for (int w = 0; w < w2; w++)
				{
					err[0] = filter_error.arr()[h * w2 + w];
					Tensor::subregion(_filter_input, &input_slice, h * _stride, w * _stride, _extent, _extent);
					TensorOperator::instance().full_gradient_s(_filter_input->arr(), err, gradient_slice.arr(), 1, _extent * _extent);
					gradient.push_back(&gradient_slice);
				}
			}

			TensorOperator::instance().m_reduce(p_gradient_map[_W[d * d1 + id]->get_id()].arr(), gradient.arr(), w2 * h2, _extent * _extent);
			gradient.reset_index();
		}
	}

	Tensor*	 delta_in = nullptr;

	if (!_input_layer.empty())
	{
		Tensor filter_delta({ _extent, _extent }, Tensor::ZERO);
		Tensor delta({ h1 + 2 * _padding, w1 + 2 * _padding }, Tensor::ZERO);
		Tensor delta_padding({ h1, w1 }, Tensor::ZERO);

		delta_in = NeuronOperator::init_auxiliary_parameter(delta_in, d1, h1, w1);
		delta_in->reset_index();

		for (int id = 0; id < d1; id++)
		{
			for (int d = 0; d < d2; d++)
			{
				Tensor filter_error = delta_out->slice(d);

				for (int h = 0; h < h2; h++)
				{
					for (int w = 0; w < w2; w++)
					{
						err[0] = filter_error.arr()[h * w2 + w];
						TensorOperator::instance().full_delta_s(filter_delta.arr(), err, _W[d * d1 + id]->get_data()->arr(), 1, _extent * _extent);
						Tensor::add_subregion(&delta, &filter_delta, h * _stride, w * _stride);
					}
				}
			}
			Tensor::subregion(&delta_padding, &delta, _padding, _padding, w1, h1);
			delta_in->push_back(&delta_padding);
			delta.fill(0);
		}

		int index = 0;

		for (auto it : _input_layer)
		{
			p_delta_map[it->get_id()] = NeuronOperator::init_auxiliary_parameter(p_delta_map[it->get_id()], _batch_size, it->get_dim());
			delta_in->splice(index, p_delta_map[it->get_id()]);
			index += it->get_dim();
		}

		delete delta_in;
	}

	delete delta_out;
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
