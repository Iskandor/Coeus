#include "ConvOperator.h"
#include "TensorOperator.h"

using namespace Coeus;

ConvOperator::ConvOperator(const int p_dim, const ACTIVATION p_activation) : NeuronOperator(p_dim, p_activation)
{
}

ConvOperator::ConvOperator(ConvOperator& p_copy, const bool p_clone) : NeuronOperator(p_copy, p_clone)
{
}

ConvOperator::~ConvOperator()
= default;

void ConvOperator::integrate(Tensor* p_dim_tensor, const int p_rows, const int p_cols, Tensor* p_input, Tensor* p_weights)
{
	_net = init_auxiliary_parameter(_net, p_dim_tensor->at(0), p_dim_tensor->at(1), p_dim_tensor->at(2));

	TensorOperator::instance().MM_prod(p_weights->arr(), true, p_input->arr(), true, _net->arr(), p_dim_tensor->at(0), p_rows, p_cols);
}

void ConvOperator::activate()
{
	TensorOperator::instance().Mv_add(_net->arr(), _bias->get_data()->arr(), _net->arr(), _net->shape(0), _net->shape(1) * _net->shape(2));
	_dnet = init_auxiliary_parameter(_dnet, 1, _dim);

	_output = _activation_function->forward(_net);
	_dnet->override(_net);
	_net->fill(0);
}

void ConvOperator::integrate(Tensor* p_input, Tensor* p_weights)
{
}


