#pragma once
#include "Tensor.h"
#include <map>
#include "Gradient.h"

namespace Coeus
{
	class __declspec(dllexport) GradientAccumulator
	{
	public:
		explicit GradientAccumulator(Gradient& p_buffer);
		GradientAccumulator(const GradientAccumulator &p_copy);
		GradientAccumulator& operator = (const GradientAccumulator& p_copy);
		~GradientAccumulator();

		void clear();

		GradientAccumulator operator + (const GradientAccumulator& p_accumulator) const;
		GradientAccumulator& operator += (const GradientAccumulator& p_accumulator);

	private:
		Gradient _gradient;
	};
}


