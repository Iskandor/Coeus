#include "CountModule.h"

using namespace Coeus;

CountModule::CountModule(const int p_state_space_size)
{
	_state = nullptr;
	_lookup_table = new int[p_state_space_size];

	for(int i = 0; i < p_state_space_size; i++)
	{
		_lookup_table[i] = 0;
	}
}

CountModule::~CountModule()
{
	delete _lookup_table;
}

void CountModule::update(Tensor* p_state)
{
	_lookup_table[p_state->max_value_index()]++;
	_state = p_state;
}

float CountModule::uncertainty_motivation()
{
	const float reward = 1.f / sqrt(_lookup_table[_state->max_value_index()]);
	return reward;
}

float CountModule::familiarity_motivation()
{
	const float reward = tanh(_lookup_table[_state->max_value_index()] / 100);
	return reward;
}

float CountModule::intermediate_novelty_motivation(float p_sigma)
{
	return 0;
}

float CountModule::surprise_motivation()
{
	return 0;
}

float CountModule::progress_uncertainty_motivation()
{
	return 0;
}

float CountModule::progress_familiarity_motivation()
{
	return 0;
}
