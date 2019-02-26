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
}

void PPLBatchModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	auto start = chrono::high_resolution_clock::now();

	for (auto& it : _gradient)
	{
		_gradient[it.first].fill(0);
	}

	critical_section mutex;

	parallel_for(0, p_batch, [&](const int i) {
		size_t index = p_b * p_batch + i;
		if (index >= p_input->size()) index = i;
		cout << index << endl;

		_network_gradient[i]->activate(p_input->at(index));
		Tensor dloss = _cost_function->cost_deriv(_clone_network[i]->get_output(), p_target->at(index));
		_network_gradient[i]->calc_gradient(&dloss);
	},
	static_partitioner()
	);

	auto end = chrono::high_resolution_clock::now();

	//cout << "Gradient" << p_b << ": " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;

	start = chrono::high_resolution_clock::now();

	//parallel_reduce(begin(words), end(words), wstring())

	for(int i = 0; i < _batch_size; i++)
	{
		for (auto& it : *_network_gradient[i]->get_gradient())
		{
			_gradient[it.first] += it.second;
		}
	}

	end = chrono::high_resolution_clock::now();

	//cout << "Accumulate" << p_b << ": " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
}

float PPLBatchModule::get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	float error = 0;
	int nbatch = p_input->size() / _batch_size;
	if (p_input->size() % _batch_size > 0) nbatch++;
	int size = _batch_size;

	critical_section mutex;

	for (int b = 0; b < nbatch; b++)
	{
		if (b * _batch_size + _batch_size > p_input->size())
		{
			size = p_input->size() - b * _batch_size;
		}

		parallel_for(0, size, [&](const int i) {
			const int index = b * _batch_size + i;
			_clone_network[i]->activate(p_input->at(index));
			const float e = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(index));
			
			mutex.lock();
			error += e;
			mutex.unlock();

		},
		static_partitioner()
		);
	}

	return error;
}