#include "AnnealingScheduler.h"
#include <algorithm>
#include <iostream>

using namespace Coeus;

AnnealingScheduler::AnnealingScheduler(const int p_model_size):
	_model_size(p_model_size),
	_step(0)
{
}

AnnealingScheduler::~AnnealingScheduler()
= default;

float AnnealingScheduler::get_alpha()
{
	_step++;
	const float alpha = 1.0f / sqrt(_model_size) * std::min<float>(1.0f / sqrt(_step), _step * _warmup);

	//std::cout << alpha << std::endl;

	return alpha;
}
