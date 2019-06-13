#include "IActivationFunction.h"
#include "NeuronOperator.h"

using namespace Coeus;

IActivationFunction::IActivationFunction(const ACTIVATION p_id): _input(nullptr), _output(nullptr), _type(p_id)
{
}

IActivationFunction::~IActivationFunction()
{
	delete _input;
	delete _output;
}

Tensor* IActivationFunction::forward(Tensor* p_input)
{
	if (p_input->rank() == 1)
	{
		_input = NeuronOperator::init_auxiliary_parameter(_input, 1, p_input->shape(0));
	}
	if (p_input->rank() == 2)
	{
		_input = NeuronOperator::init_auxiliary_parameter(_input, p_input->shape(0), p_input->shape(1));
	}
	if (p_input->rank() == 3)
	{
		_input = NeuronOperator::init_auxiliary_parameter(_input, p_input->shape(0), p_input->shape(1), p_input->shape(2));
	}

	_input->override(p_input);

	if (p_input->rank() == 1)
	{
		_output = NeuronOperator::init_auxiliary_parameter(_output, 1, p_input->shape(0));
	}
	if (p_input->rank() == 2)
	{
		_output = NeuronOperator::init_auxiliary_parameter(_output, p_input->shape(0), p_input->shape(1));
	}
	if (p_input->rank() == 3)
	{
		_output = NeuronOperator::init_auxiliary_parameter(_output, p_input->shape(0), p_input->shape(1), p_input->shape(2));
	}

	return _output;
}

json IActivationFunction::get_json()
{
	json data;

	data["type"] = _type;

	return data;
}
