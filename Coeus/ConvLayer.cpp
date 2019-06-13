#include "ConvLayer.h"
#include "IDGen.h"
#include "ActivationFunctionFactory.h"

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

	_y = new NeuronOperator*[p_filters];
	_W = new Param*[p_filters];

	for(int i = 0; i < p_filters; i++)
	{
		_y[i] = new NeuronOperator(1, LINEAR);
		_W[i] = new Param(IDGen::instance().next(), new Tensor({ 1, _extent * _extent }, Tensor::ZERO));
		_initializer->init(_W[i]->get_data());
		add_param(_W[i]->get_id(), _W[i]->get_data());
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
					_y[f]->integrate(_filter_input, _W[f]->get_data());
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

void ConvLayer::calc_delta(map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
}

void ConvLayer::calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map)
{
	Tensor*	 delta_out = _activation_function->backward(p_delta_map[_id]);



	delete delta_out;
}

void ConvLayer::override(BaseLayer* p_source)
{
}

void ConvLayer::reset()
{
}