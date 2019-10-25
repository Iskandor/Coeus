#include "GradientAccumulator.h"
#include "TensorOperator.h"

using namespace Coeus;

GradientAccumulator::GradientAccumulator(map<string, Tensor>& p_buffer)
{
	_gradient = p_buffer;
}

GradientAccumulator::GradientAccumulator(const GradientAccumulator& p_copy)
{
	_gradient = p_copy._gradient;
}

GradientAccumulator& GradientAccumulator::operator=(const GradientAccumulator& p_copy)
= default;

GradientAccumulator::~GradientAccumulator()
= default;

void GradientAccumulator::clear() const
{
	for (auto& it : _gradient)
	{
		it.second.fill(0);
	}
}

GradientAccumulator GradientAccumulator::operator+(const GradientAccumulator& p_accumulator) const
{
	GradientAccumulator temp(*this);

	return temp += p_accumulator;
}

GradientAccumulator& GradientAccumulator::operator+=(const GradientAccumulator& p_accumulator)
{
	for (auto& it : p_accumulator._gradient)
	{		
		 TensorOperator::instance().vv_add(_gradient[it.first].arr(), it.second.arr(), _gradient[it.first].arr(), _gradient[it.first].size());
	}

	return *this;
}
