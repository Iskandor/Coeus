#include "PPLBatchModule.h"
#include <ppl.h>

using namespace Coeus;
using namespace Concurrency;

PPLBatchModule::PPLBatchModule(NeuralNetwork* p_network, ICostFunction* p_cost_function, const int p_batch) : IBatchModule(p_batch),
	_cost_function(p_cost_function), 
	_network(p_network)
{
	_clone_network = new NeuralNetwork*[p_batch];
	_network_gradient = new NetworkGradient*[p_batch];
	_error = new double[p_batch];

	for(int i = 0; i < p_batch; i++)
	{
		_clone_network[i] = p_network->clone();
		_network_gradient[i] = new NetworkGradient(_clone_network[i]);
	}

	_gradient = _network_gradient[0]->get_empty_params();
}


PPLBatchModule::~PPLBatchModule()
{
	for (int i = 0; i < _batch_size; i++)
	{
		delete _clone_network[i];
		delete _network_gradient[i];
	}

	delete _network_gradient;
	delete _clone_network;
	delete _error;
}

double PPLBatchModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	double error = 0;
	
	//auto start = chrono::high_resolution_clock::now();

	for (auto& it : _gradient)
	{
		_gradient[it.first].fill(0);
	}

	parallel_for(0, p_batch, [&](const int i) {
		const int index = p_b * p_batch + i;
		_clone_network[i]->calc_partial_derivs(p_input->at(index));
		Tensor dloss = _cost_function->cost_deriv(_clone_network[i]->get_output(), p_target->at(index));
		_error[i] = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(i));
		_network_gradient[i]->calc_gradient(&dloss);		
	},
	static_partitioner()
	);

	for(int i = 0; i < _batch_size; i++)
	{
		for (auto& it : *_network_gradient[i]->get_gradient())
		{
			_gradient[it.first] += it.second;
		}
		error += _error[i];
	}

	//auto end = chrono::high_resolution_clock::now();

	//cout << "Batch" << p_b << ": " << (end - start).count() * ((double)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;

	return error;
}
