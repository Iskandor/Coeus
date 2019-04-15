#pragma once
#include "Tensor.h"
#include <map>

using namespace FLAB;

namespace Coeus
{
	class __declspec(dllexport) GradientAccumulator
	{
	public:
		explicit GradientAccumulator(map<string, Tensor>* p_gradient);
		GradientAccumulator(const GradientAccumulator &p_copy);
		GradientAccumulator& operator = (const GradientAccumulator& p_copy);
		~GradientAccumulator();

		void clear() const;

		GradientAccumulator operator + (const GradientAccumulator& p_accumulator) const;
		GradientAccumulator& operator += (const GradientAccumulator& p_accumulator);

	private:
		map<string, Tensor>* _gradient;
	};
}


