#include "Encoder.h"
#include <cassert>
#include "Metrics.h"

using namespace Coeus;

Encoder::Encoder()
{
}


Encoder::~Encoder()
{
}

void Encoder::one_hot(Tensor& p_result, const int p_value)
{
	if (p_result.size() <= p_value)
	{
		assert(("Encoder: one hot tensor too small", 0));
	}

	p_result.fill(0);
	p_result[p_value] = 1;
}

void Encoder::pop_code(Tensor& p_result, const double p_value, const double p_lower_limit, const double p_upper_limit)
{
	if (p_lower_limit > p_value || p_upper_limit < p_value)
	{
		assert(("Encoder: pop code value out of range", 0));
	}

	double delta = (p_upper_limit - p_lower_limit) / (p_result.size() - 1);

	for(int i = 0; i < p_result.size(); i++)
	{
		p_result[i] = Metrics::gaussian_distance(p_value, 0.4, p_lower_limit + i * delta);
	}
}
