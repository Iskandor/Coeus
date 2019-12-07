#pragma once
#include "Tensor.h"
#include <vector>
#include <map>
#include "GradientAccumulator.h"

namespace Coeus {
	class __declspec(dllexport) IBatchModule
	{
	public:
		IBatchModule(int p_batch);
		virtual ~IBatchModule();

		virtual void run_batch(int p_b, int p_batch, vector<Tensor*>* p_input, vector<Tensor*>* p_target) = 0;
		virtual float get_error(vector<Tensor*>* p_input, vector<Tensor*>* p_target) = 0;
		Gradient& get_gradient() { return _gradient; }
		int get_batch_size() const { return _batch_size; }

	protected:
		int					 _batch_size;
		Gradient			 _gradient;
		GradientAccumulator* _gradient_accumulator;

	};
}
