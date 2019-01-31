#include "Encoder.h"
#include <cassert>
#include "Metrics.h"
#include <bitset>

using namespace Coeus;

Encoder::Encoder()
{
}


Encoder::~Encoder()
{
}

void Encoder::one_hot(int* p_result, const int p_size, const int p_value)
{
	for(int i = 0; i < p_size; i++)
	{
		p_result[i] = 0;
	}

	p_result[p_value] = 1;
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

	const double c = (p_result.size() - 1) / (p_upper_limit - p_lower_limit);

	for(int i = 0; i < p_result.size(); i++)
	{
		p_result[i] = Metrics::gaussian_distance((p_value - p_lower_limit) * c, 0.4, i);
	}
}

void Encoder::grey_code(Tensor& p_result, Tensor& p_bin)
{
	p_result[0] = p_bin[0];

	for (int i = 1; i < p_bin.size(); i++) {
		p_result[i] = xor_c(p_bin[i - 1], p_bin[i]);
	}
}
