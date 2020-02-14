#include "DiscreteExploration.h"
#include "RandomGenerator.h"

DiscreteExploration::DiscreteExploration(const METHOD p_method, const float p_exploration_parameter, Coeus::IInterpolation* p_interpolation) :
	_method(p_method),
	_param(p_exploration_parameter),
	_interpolation(p_interpolation)
{
	
}

DiscreteExploration::~DiscreteExploration()
{
	delete _interpolation;
}

Tensor DiscreteExploration::explore(Tensor* p_values) const
{
	Tensor output({ p_values->size() }, Tensor::ZERO);
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

void DiscreteExploration::update(int p_timestep)
{
	if (_interpolation != nullptr) _param = _interpolation->interpolate(p_timestep);
}

void DiscreteExploration::explore_egreedy(Tensor& p_output, Tensor* p_values) const
{
	int action;
	const float random = RandomGenerator::get_instance().random();

	if (random < _param) {
		action = RandomGenerator::get_instance().random(0, p_values->size() - 1);
	}
	else {
		action = p_values->max_value_index();
	}

	p_output.fill(0);
	p_output[action] = 1;
}

void DiscreteExploration::explore_boltzman(Tensor& p_output, Tensor* p_values) const
{
	Tensor evals = Tensor({ p_values->size() }, Tensor::ZERO);
	
	float sum = 0;

	for (int i = 0; i < p_values->size(); i++)
	{
		evals[i] = exp((*p_values)[i] / _param);
		sum += evals[i];
	}

	evals /= sum;

	const int action = RandomGenerator::get_instance().choice(evals.arr(), evals.size());

	p_output.fill(0);
	p_output[action] = 1;
}
