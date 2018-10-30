#include "BaseGradientAlgorithm.h"

using namespace Coeus;

BaseGradientAlgorithm::BaseGradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_cost_function = nullptr;	
	_update_rule = nullptr;
	_parallel_module = nullptr;
	_batch = 0;
}


BaseGradientAlgorithm::~BaseGradientAlgorithm()
{
	delete _update_rule;
	delete _cost_function;
	delete _network_gradient;
	delete _parallel_module;
}

double BaseGradientAlgorithm::train(Tensor* p_input, Tensor* p_target) {
	_network->activate(p_input);

	double error = 0;

	if (p_target != nullptr)
	{
		error = train(p_target);
	}
	//_network_gradient->check_gradient(p_input, p_target);

	return error;
}

double BaseGradientAlgorithm::train(vector<Tensor*>* p_input, Tensor* p_target)
{
	_network->activate(p_input);
	const double error = train(p_target);

	//_network_gradient->check_gradient(p_input, p_target);

	return error;
}

double BaseGradientAlgorithm::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch) {
	double error = 0;

	_batch = p_batch;

	int nbatch = p_input->size() / _batch;

	if (p_input->size() % _batch > 0)
	{
		nbatch++;
	}

	if (_parallel_module == nullptr)
	{
		_parallel_module = new ParallelModule(_network, _update_rule, _cost_function, _batch);
	}


	for (int b = 0; b < nbatch; b++)
	{
		if (p_input->size() < b * _batch + _batch)
		{
			_batch = p_input->size() - b * _batch;
		}

		error += _parallel_module->run_batch(b, _batch, p_input, p_target);

		_network->update(_parallel_module->get_update());
	}
	
	/*
	for (int b = 0; b < nbatch; b++)
	{
		if (p_input->size() < b * _batch + _batch)
		{
			_batch = p_input->size() - b * _batch;
		}

		for (int i = 0; i < _batch; i++) {
			const int index = b * _batch + i;
			_network->activate(p_input->at(index));
			error += _cost_function->cost(_network->get_output(), p_target->at(index));
			_network_gradient->calc_gradient(p_target->at(index));
			_update_rule->calc_update();
		}

		for (auto it = _update.begin(); it != _update.end(); ++it) {
			_update_batch[it->first] /= _batch;
		}

		_network->update(_update_batch);
	}
	*/

	return error;
}

double BaseGradientAlgorithm::train(Tensor* p_target) const
{
	const double error = _cost_function->cost(_network->get_output(), p_target);

	_network_gradient->calc_gradient(p_target);
	_update_rule->calc_update();
	_network->update(_update_rule->get_update());

	return error;
}

void BaseGradientAlgorithm::init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule)
{
	_network_gradient->init(p_cost_function);
	_cost_function = p_cost_function;
	_update_rule = p_update_rule;

}