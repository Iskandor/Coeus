#include "linear_operator.h"
#include "tensor_operator_cpu.h"

linear_operator::linear_operator(param* p_weights, param* p_bias):
	_input(nullptr),
	_weights(p_weights),
	_bias(p_bias),
	_gpu_flag(false)
{
}

linear_operator::~linear_operator()
= default;

tensor& linear_operator::forward(tensor& p_input)
{
	_input = &p_input;
	_output.resize({ p_input.shape(0), _weights->params().shape(1) });
	tensor_operator_cpu::mul(p_input.data(), false, _weights->params().data(), false, _output.data(), p_input.shape(0), p_input.shape(1), _weights->params().shape(1));
	tensor_operator_cpu::add(_output.data(), _output.size(), _bias->params().data(), _bias->params().size(), _output.data());
	return _output;
}

tensor& linear_operator::backward(tensor& p_delta)
{
	_delta.resize({ p_delta.shape(0), _weights->params().shape(0) });

	_weights->gradient().fill(0);

	tensor_operator_cpu::mul(p_delta.data(), false, _weights->params().data(), true, _delta.data(), p_delta.shape(0), p_delta.shape(1), _weights->params().shape(0));
	tensor_operator_cpu::mul(_input->data(), true, p_delta.data(), false, _weights->gradient().data(), _weights->params().shape(0), p_delta.shape(0), _weights->params().shape(1));
	tensor_operator_cpu::reduce_sum(p_delta.data(), p_delta.shape(0), _bias->gradient().data(), _bias->gradient().size());

	return _delta;
}
