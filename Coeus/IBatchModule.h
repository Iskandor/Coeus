#pragma once
#include "Tensor.h"
#include <vector>
#include <map>

using namespace FLAB;

namespace Coeus {
	class __declspec(dllexport) IBatchModule
	{
	public:
		IBatchModule(int p_batch);
		virtual ~IBatchModule();

		virtual double run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) = 0;
		map<string, Tensor>* get_update() { return &_update_batch; }
		int get_batch_size() const { return _batch_size; }

	protected:
		int					_batch_size;
		map<string, Tensor> _update_batch;

	};
}

