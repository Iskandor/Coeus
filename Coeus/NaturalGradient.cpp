/*
 * Natural Gradient algorithm
 * https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
 */
#include "NaturalGradient.h"
#include "TensorOperator.h"

using namespace Coeus;

NaturalGradient::NaturalGradient(NeuralNetwork* p_network) : NetworkGradient(p_network),
    _fim_n(1),
	_epsilon(1e-8f)
{
	for (auto it = p_network->_params.begin(); it != p_network->_params.end(); ++it)
	{
		_fim[it->first] = Tensor({it->second->shape(0), it->second->shape(0)}, Tensor::ZERO);
		_inv_fim[it->first] = Tensor({it->second->shape(0), it->second->shape(0)}, Tensor::ONES);
		_cache[it->first] = Tensor({it->second->shape(0), it->second->shape(0)}, Tensor::ZERO);
	}
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

	_fim_n++;

	for (auto& it : _gradient)
	{
		//_invFisherMatrix[connectionId] = (1 + _epsilon) * _invFisherMatrix[connectionId] - _epsilon * _invFisherMatrix[connectionId] * _regGradient[connectionId] * _regGradient[connectionId].T() * _invFisherMatrix[connectionId];
		const int rows = it.second.shape(0);
		const int cols = it.second.shape(1);
		
		// estimate Fisher information matrix (FIM)
		TensorOperator::instance().MM_prod(it.second.arr(), false, it.second.arr(), true, static_cast<float>(_fim_n - 1) / _fim_n, _fim[it.first].arr(), 1.f / _fim_n, rows, cols, rows);

		// estimate inverse FIM
		TensorOperator::instance().MM_prod(_inv_fim[it.first].arr(), false, _fim[it.first].arr(), false, _cache[it.first].arr(), rows, rows, rows, false);
		TensorOperator::instance().MM_prod(_cache[it.first].arr(), false, _inv_fim[it.first].arr(), false, _cache[it.first].arr(), rows, rows, rows, false);
		TensorOperator::instance().vv_sub(_inv_fim[it.first].arr(), (1 + _epsilon), _cache[it.first].arr(), _epsilon, _inv_fim[it.first].arr(), rows * rows);

		// calculate inverse FIM
		//TensorOperator::instance().inv_M(_fim[it.first].arr(), _inv_fim[it.first].arr(), rows, rows);
		//cout << _inv_fim[it.first] << endl;
		// use FIM inverse to calculate natural gradient
		TensorOperator::instance().MM_prod(_inv_fim[it.first].arr(), false, it.second.arr(), false, it.second.arr(), rows, rows, cols, false);
	}	
}

void NaturalGradient::calc_gradient(vector<Tensor*>* p_input, Tensor* p_loss)
{
	
}
