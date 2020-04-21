#include "discrete_exploration.h"
#include "random_generator.h"

discrete_exploration::discrete_exploration(const METHOD p_method, const float p_exploration_parameter, iinterpolation* p_interpolation) :
	_method(p_method),
	_param(p_exploration_parameter),
	_interpolation(p_interpolation)
{
	
}

discrete_exploration::~discrete_exploration()
{
	delete _interpolation;
}

tensor discrete_exploration::explore(tensor& p_values) const
{
	tensor output({ p_values.size() }, tensor::ZERO);
	switch(_method)
	{
	case EGREEDY:
		explore_egreedy(output, p_values);
		break;
	case BOLTZMAN:
		explore_boltzman(output, p_values);
		break;
	default: ;
	}

	return output;
}

void discrete_exploration::update(int p_timestep)
{
	if (_interpolation != nullptr) _param = _interpolation->interpolate(p_timestep);
}

void discrete_exploration::explore_egreedy(tensor& p_output, tensor& p_values) const
{
	int action;
	const float random = random_generator::instance().random();

	if (random < _param) {
		action = random_generator::instance().random(0, p_values.size() - 1);
	}
	else {
		action = p_values.max_index()[0];
	}

	p_output.fill(0);
	p_output[action] = 1;
}

void discrete_exploration::explore_boltzman(tensor& p_output, tensor& p_values) const
{
	tensor evals = tensor({ p_values.size() }, tensor::ZERO);
	
	float sum = 0;

	for (int i = 0; i < p_values.size(); i++)
	{
		evals[i] = exp(p_values[i] / _param);
		sum += evals[i];
	}

	evals /= sum;

	const int action = random_generator::instance().choice(evals.data(), evals.size());

	p_output.fill(0);
	p_output[action] = 1;
}
