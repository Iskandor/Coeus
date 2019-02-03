#include "GradientAlgorithm.h"
#include "SingleBatchModule.h"
#include <chrono>
#include "OpenMPBatchModule.h"

using namespace Coeus;

GradientAlgorithm::GradientAlgorithm(NeuralNetwork* p_network)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_cost_function = nullptr;	
	_update_rule = nullptr;
	_batch_module = nullptr;
	_learning_rate_module = nullptr;
}


GradientAlgorithm::~GradientAlgorithm()
{
	delete _update_rule;
	delete _cost_function;
	delete _network_gradient;
	delete _batch_module;
	delete _learning_rate_module;
}

double GradientAlgorithm::train(Tensor* p_input, Tensor* p_target)
{
	double error = 0;
	double alpha = 0;
	
	if (_learning_rate_module != nullptr) {
		alpha = _learning_rate_module->get_alpha();
	}

	if (p_target->rank() == 1)
	{
		_network_gradient->calc_partial_derivs(p_input);
		Tensor dloss = _cost_function->cost_deriv(_network->get_output(), p_target);

		_network_gradient->calc_gradient(&dloss);
		error = _cost_function->cost(_network->get_output(), p_target);	
		_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
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

			_network_gradient->calc_partial_derivs(&input);
			Tensor dloss = _cost_function->cost_deriv(_network->get_output(), &target);

			_network_gradient->calc_gradient(&dloss);
			error += _cost_function->cost(_network->get_output(), p_target);
			_update_rule->calc_update(_network_gradient->get_gradient(), alpha);
			_network->update(_update_rule->get_update());
		}
	}

	//_network_gradient->check_gradient(p_input, p_target);

	return error;
}

double GradientAlgorithm::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch) {

	double error = 0;
	double alpha = 0;

	if (_learning_rate_module != nullptr) {
		alpha = _learning_rate_module->get_alpha();
	}

	for(unsigned int i = 0; i < p_input->size(); i++)
	{
		_network->activate((*p_input)[i]);
		error += _cost_function->cost(_network->get_output(), (*p_target)[i]);
	}


	int nbatch = p_input->size() / p_batch;

	if (p_input->size() % p_batch > 0)
	{
		nbatch++;
	}

	if (_batch_module == nullptr)
	{
		//_batch_module = new SingleBatchModule(_network, _network_gradient, _cost_function, p_batch);
		_batch_module = new PPLBatchModule(_network, _cost_function, p_batch);
		//_batch_module = new OpenMPBatchModule(_network, _cost_function, p_batch);
	}
	else if (_batch_module->get_batch_size() < p_batch)
	{
		delete _batch_module;
		//_batch_module = new SingleBatchModule(_network, _network_gradient, _cost_function, p_batch);
		_batch_module = new PPLBatchModule(_network, _cost_function, p_batch);
		//_batch_module = new OpenMPBatchModule(_network, _cost_function, p_batch);
	}

	for (int b = 0; b < nbatch; b++)
	{
		if (b * p_batch + p_batch > p_input->size())
		{
			p_batch = p_input->size() - b * p_batch;
		}

		_batch_module->run_batch(b, p_batch, p_input, p_target);

		_update_rule->calc_update(_batch_module->get_gradient(), alpha);
		_network->update(_update_rule->get_update());
		
		//cout << b << "/" << nbatch << "  ";
	}
	//cout << endl;
	
	return error;
}

void GradientAlgorithm::add_learning_rate_module(ILearningRateModule* p_learning_rate_module)
{
	_learning_rate_module = p_learning_rate_module;
}

void GradientAlgorithm::init(ICostFunction* p_cost_function, IUpdateRule* p_update_rule)
{
	_cost_function = p_cost_function;
	_update_rule = p_update_rule;
}
