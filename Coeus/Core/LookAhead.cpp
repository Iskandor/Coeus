#include "LookAhead.h"
#include "RuleFactory.h"

using namespace Coeus;

LookAhead::LookAhead(NeuralNetwork* p_network) : GradientAlgorithm(p_network), _k(0), _kt(0), _error(0)
{
	_slow_params = p_network->get_params();
}

LookAhead::~LookAhead()
= default;

float LookAhead::train(Tensor* p_input, Tensor* p_target)
{
	_kt++;
	const float error = GradientAlgorithm::train(p_input, p_target);

	slow_update();

	return error;
}

float LookAhead::train(vector<Tensor*>* p_input, Tensor* p_target)
{
	_kt++;
	const float error = GradientAlgorithm::train(p_input, p_target);

	slow_update();

	return error;
}

float LookAhead::train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, bool p_update)
{
	_kt++;
	const float error = GradientAlgorithm::train(p_input, p_target, p_update);

	slow_update();

	return error;
}

void LookAhead::init(GRADIENT_RULE p_update_rule, ICostFunction* p_cost_function, const float p_alpha, const int p_k)
{
	GradientAlgorithm::init(p_cost_function, RuleFactory::create_rule(p_update_rule, _network, p_alpha));
	_k = p_k;
}

void LookAhead::slow_update()
{
	if (_kt == _k)
	{
		for (auto p : _slow_params)
		{
			p.second += 0.5 * (*(*_network)[p.first] - p.second);
		}
		_network->override(_slow_params);
		_kt = 0;
	}
}
