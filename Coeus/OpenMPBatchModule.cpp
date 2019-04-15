#include "OpenMPBatchModule.h"
#include <chrono>
#include <omp.h>
#include "TensorOperator.h"

using namespace Coeus;

OpenMPBatchModule::OpenMPBatchModule(NeuralNetwork* p_network, ICostFunction* p_cost_function, int p_batch) : IBatchModule(p_batch),
	_cost_function(p_cost_function),
	_network(p_network)
{
	_clone_network = new NeuralNetwork*[p_batch];
	_network_gradient = new NetworkGradient*[p_batch];
	_error = new float[p_batch];

	for (int i = 0; i < p_batch; i++)
	{
		_clone_network[i] = p_network->clone();
		_network_gradient[i] = new NetworkGradient(_clone_network[i]);
	}

	_gradient = _network->get_empty_params();
}

OpenMPBatchModule::~OpenMPBatchModule()
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

void OpenMPBatchModule::run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	auto start = chrono::high_resolution_clock::now();

	#pragma omp parallel for
	for (int i = 0; i < p_batch; i++) {
		size_t index = p_b * p_batch + i;
		if (index >= p_input->size()) index = i;

		_network_gradient[i]->activate(p_input->at(index));
		Tensor dloss = _cost_function->cost_deriv(_clone_network[i]->get_output(), p_target->at(index));
		_network_gradient[i]->calc_gradient(&dloss);
	}

	auto end = chrono::high_resolution_clock::now();

	//cout << "Gradient" << p_b << ": " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;

	start = chrono::high_resolution_clock::now();

	#pragma omp parallel
	{
		_gradient_accumulator->clear();
		#pragma omp for nowait
		for (int i = 0; i < _batch_size; i++)
		{
			for (auto it = _network_gradient[i]->get_gradient()->begin(); it != _network_gradient[i]->get_gradient()->end(); ++it) {
				#pragma omp critical
				TensorOperator::instance().vv_add(_gradient[it->first].arr(), it->second.arr(), _gradient[it->first].arr(), _gradient[it->first].size());
			}
		}
	}

	end = chrono::high_resolution_clock::now();

	//cout << "Accumulate" << p_b << ": " << (end - start).count() * ((float)chrono::high_resolution_clock::period::num / chrono::high_resolution_clock::period::den) << endl;
}

float OpenMPBatchModule::get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target)
{
	float error = 0;
	int nbatch = p_input->size() / _batch_size;
	if (p_input->size() % _batch_size > 0) nbatch++;
	int size = _batch_size;

	omp_lock_t writelock;

	omp_init_lock(&writelock);

	for (int b = 0; b < nbatch; b++)
	{
		if (b * _batch_size + _batch_size > p_input->size())
		{
			size = p_input->size() - b * _batch_size;
		}

		#pragma omp parallel for
		for(int i = 0; i < size; i++) {
			const int index = b * _batch_size + i;
			_clone_network[i]->activate(p_input->at(index));
			const float e = _cost_function->cost(_clone_network[i]->get_output(), p_target->at(index));
			omp_set_lock(&writelock);
			error += e;
			omp_unset_lock(&writelock);
		}
	}

	omp_destroy_lock(&writelock);

	return error;
}
