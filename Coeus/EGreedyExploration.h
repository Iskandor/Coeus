#pragma once
#include "Tensor.h"
#include "IInterpolation.h"

using namespace FLAB;

namespace Coeus
{
	class __declspec(dllexport) EGreedyExploration
	{
	public:
		EGreedyExploration(float p_epsilon, IInterpolation* p_interpolation);
		~EGreedyExploration();

		int get_action(Tensor* p_values) const;
		void update(int p_t);

	private:
		float			_epsilon;
		IInterpolation *_interpolation;
	};
}