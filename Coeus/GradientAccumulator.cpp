#include "GradientAccumulator.h"
#include "TensorOperator.h"

using namespace Coeus;

GradientAccumulator::GradientAccumulator(Gradient& p_buffer)
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

void GradientAccumulator::clear()
{
	_gradient.fill(0);
}

GradientAccumulator GradientAccumulator::operator+(const GradientAccumulator& p_accumulator) const
{
	GradientAccumulator temp(*this);

	return temp += p_accumulator;
}

GradientAccumulator& GradientAccumulator::operator+=(const GradientAccumulator& p_accumulator)
{
	_gradient += p_accumulator._gradient;

	return *this;
}
