/*
 * Natural gradient algorithm
 * https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
 */
#include "NaturalGradient.h"
#include "TensorOperator.h"

using namespace Coeus;

NaturalGradient::NaturalGradient(NeuralNetwork* p_network) : NetworkGradient(p_network),
	_epsilon(1e-2f)
{
	for (auto& _param : p_network->_params)
	{
		const int row = _param.second->shape(0) * _param.second->shape(1);
		_fim[_param.first] = Tensor({ row, row }, Tensor::ONES);
		_inv_fim[_param.first] = Tensor({ row, row }, Tensor::ZERO);
	}

	_cache = p_network->get_empty_params();
}


NaturalGradient::~NaturalGradient()
= default;

void NaturalGradient::calc_gradient(Tensor* p_loss) {

	BaseLayer* output_layer = _network->_layers[_network->_output_layer];
	output_layer->set_delta_out(p_loss);

	for (auto& it : _calculation_graph)
	{
		it->calc_gradient(_cache, _derivative);
	}
	
	for (auto& it : _gradient)
	{
		const int size = it.second.shape(0) * it.second.shape(1);

		Tensor temp = _fim[it.first];
		
		// estimate Fisher information matrix (FIM)
		TensorOperator::instance().MM_prod(_cache[it.first].arr(), false, _cache[it.first].arr(), true, temp.arr(), size, 1, size);
		TensorOperator::instance().vv_sub(temp.arr(), _fim[it.first].arr(), temp.arr(), size * size);
		TensorOperator::instance().vv_add(_fim[it.first].arr(), 1, temp.arr(), _epsilon, _fim[it.first].arr(), size * size);		

		// calculate inverse FIM
		TensorOperator::instance().inv_M(_fim[it.first].arr(), _inv_fim[it.first].arr(), size, size);
		
		// use FIM inverse to calculate natural gradient
		TensorOperator::instance().MM_prod(_inv_fim[it.first].arr(), false, _cache[it.first].arr(), false, it.second.arr(), size, size, 1);
	}
	if (_epsilon > 1e-8)
	{
		_epsilon *= 0.995;
	}	
}

void NaturalGradient::calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss)
{
	
}
