#include "NaturalGradient.h"
#include "TensorOperator.h"
#include <omp.h>

using namespace Coeus;

NaturalGradient::NaturalGradient(NeuralNetwork* p_network) : NetworkGradient(p_network),
	_epsilon(1e-10f),
	_alpha(4.f)
{
	for (auto& _param : p_network->_params)
	{
		const int row = _param.second->shape(0) * _param.second->shape(1);
		_fim[_param.first] = Tensor({ row, row }, Tensor::ZERO);
		_inv_fim[_param.first] = Tensor({ row, row }, Tensor::ZERO);
	}

	_natural_gradient.init(p_network);
}


NaturalGradient::~NaturalGradient()
= default;

void NaturalGradient::calc_gradient(Tensor* p_loss) {

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];
	output_layer->set_delta_out(p_loss);

	for (auto& it : _calculation_graph)
	{
		it->calc_gradient(_gradient, _derivative);
	}
	
	for (auto it = _gradient.begin(); it != _gradient.end(); ++it)
	{		
		const int size = it->second.shape(0) * it->second.shape(1);

		Tensor temp = _fim[it->first];
		Tensor identity({ size, size }, Tensor::ONES);

		// estimate Fisher information matrix (FIM)
		TensorOperator::instance().MM_prod(_gradient[it->first].arr(), false, _gradient[it->first].arr(), true, temp.arr(), size, 1, size);
		const float x_trace = temp.trace();
		const float beta = _alpha * max(x_trace, _epsilon) / size;
		TensorOperator::instance().vv_add(temp.arr(), 1, identity.arr(), beta, temp.arr(), size * size);
		TensorOperator::instance().vv_add(_fim[it->first].arr(), temp.arr(), _fim[it->first].arr(), size * size);		

		// calculate inverse FIM
		TensorOperator::instance().inv_M(_fim[it->first].arr(), _inv_fim[it->first].arr(), size, size);

		// use FIM inverse to calculate natural gradient
		TensorOperator::instance().MM_prod(_inv_fim[it->first].arr(), false, it->second.arr(), false, _natural_gradient[it->first].arr(), size, size, 1);
		
		TensorOperator::instance().MM_prod(_natural_gradient[it->first].arr(), false, _natural_gradient[it->first].arr(), true, temp.arr(), size, 1, size);
		const float nx_trace = temp.trace();
		const float gamma = nx_trace != 0 ? sqrt(x_trace / nx_trace) : 1;

		_natural_gradient[it->first] *= gamma;
	}
}

Gradient& NaturalGradient::get_gradient()
{
	return _natural_gradient;
}
