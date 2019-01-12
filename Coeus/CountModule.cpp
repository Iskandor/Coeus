#include "CountModule.h"

using namespace Coeus;

CountModule::CountModule(const int p_state_space_size)
{
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

double CountModule::get_reward(Tensor* p_state) const
{
	const double reward = 1 / sqrt(_lookup_table[p_state->max_value_index()]);
	return reward;
}

void CountModule::update(Tensor* p_state) const
{
	_lookup_table[p_state->max_value_index()]++;
}
