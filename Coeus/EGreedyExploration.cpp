#include "EGreedyExploration.h"
#include "RandomGenerator.h"
#include <iostream>

using namespace Coeus;

EGreedyExploration::EGreedyExploration(const float p_epsilon, IInterpolation* p_interpolation)
{
	_epsilon = p_epsilon;
	_interpolation = p_interpolation;
}


EGreedyExploration::~EGreedyExploration()
{
	delete _interpolation;
}

void EGreedyExploration::update(const int p_t)
{
	if (_interpolation != nullptr)
	{
		_epsilon = _interpolation->interpolate(p_t);
		cout << _epsilon << endl;
	}
}

int EGreedyExploration::get_action(Tensor* p_values) const
{
	int action = 0;
	const float random = RandomGenerator::get_instance().random();
	
	if (random < _epsilon) {
		action = RandomGenerator::get_instance().random(0, p_values->size() - 1);
	}
	else {
		action = p_values->max_value_index();
	}

	return action;
}
