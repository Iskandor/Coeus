#include "BaseGradientAlgorithm.h"
#include "SingleBatchModule.h"

using namespace Coeus;

BaseGradientAlgorithm::BaseGradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_cost_function = nullptr;	
	_update_rule = nullptr;
	_batch_module = nullptr;	
}


BaseGradientAlgorithm::~BaseGradientAlgorithm()
{
	delete _update_rule;
	delete _cost_function;
	delete _network_gradient;
	delete _batch_module;
}

double BaseGradientAlgorithm::train(Tensor* p_input, Tensor* p_target) const
{
	double error = 0;

	if (p_target->rank() == 1)
	{
		_network_gradient->calc_gradient(p_input, p_target);
		error = _cost_function->cost(_network->get_output(), p_target);	
		_update_rule->calc_update(_network_gradient->get_gradient());
		_network->update(_update_rule->get_update());
	}

	if (p_input->rank() == 2 && p_target->rank() == 2)
	{
		_network->reset();

		Tensor input = Tensor::Zero({ p_input->shape(1) });
		Tensor target = Tensor::Zero({ p_target->shape(1) });

		for (int i = 0; i < p_input->shape(0); i++)
		{
			p_input->get_row(input, i);
			p_target->get_row(target, i);

			_network_gradient->calc_gradient(p_input, p_target);
			error += _cost_function->cost(_network->get_output(), p_target);
			_update_rule->calc_update(_network_gradient->get_gradient());
			_network->update(_update_rule->get_update());
		}
	}

	//_network_gradient->check_gradient(p_input, p_target);

	return error;
}

double BaseGradientAlgorithm::train(vector<Tensor*>* p_input, Tensor* p_target) const
{
	/*
	_network->activate(p_input);
	const double error = train(p_target);

	//_network_gradient->check_gradient(p_input, p_target);

	return error;
	*/
	return 0;
}

double BaseGradientAlgorithm::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch) {
	double error = 0;

	int nbatch = p_input->size() / p_batch;

	if (p_input->size() % p_batch > 0)
	{
		nbatch++;
	}

	if (_batch_module == nullptr)
	{
		//_batch_module = new SingleBatchModule(_network, _network_gradient, _update_rule, _cost_function, p_batch);
		_batch_module = new PPLBatchModule(_network, _update_rule, _cost_function, p_batch);
	}
	else if (_batch_module->get_batch_size() < p_batch)
	{
		delete _batch_module;
		//_batch_module = new SingleBatchModule(_network, _network_gradient, _update_rule, _cost_function, p_batch);
		_batch_module = new PPLBatchModule(_network, _update_rule, _cost_function, p_batch);
	}

	for (int b = 0; b < nbatch; b++)
	{
		if (b * p_batch + p_batch > p_input->size())
		{
			p_batch = p_input->size() - b * p_batch;
		}

		error += _batch_module->run_batch(b, p_batch, p_input, p_target);

		_network->update(_batch_module->get_update());
	}
	
	return error;
}

void BaseGradientAlgorithm::add_learning_rate_module(ILearningRateModule* p_learning_rate_module) const
{
	_update_rule->init_learning_rate_module(p_learning_rate_module);
}

void BaseGradientAlgorithm::init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule)
{
	_network_gradient->init(p_cost_function);
	_cost_function = p_cost_function;
	_update_rule = p_update_rule;
}
