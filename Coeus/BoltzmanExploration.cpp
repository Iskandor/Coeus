#include "BoltzmanExploration.h"
#include <cmath>
#include "RandomGenerator.h"
#include <iostream>

using namespace Coeus;

BoltzmanExploration::BoltzmanExploration(const float p_T, IInterpolation* p_interpolation)
{
	_T = p_T;
	_interpolation = p_interpolation;
}

BoltzmanExploration::~BoltzmanExploration()
{
	delete _interpolation;
}

int BoltzmanExploration::get_action(Tensor* p_values)
{
	int result = 0;

	float* evals = new float[p_values->size()];
	float sum = 0;

	for(int i = 0; i < p_values->size(); i++)
	{
		evals[i] = exp((*p_values)[i] / _T);
		sum += evals[i];
	}

	for (int i = 0; i < p_values->size(); i++)
	{
		evals[i] /= sum;
	}

	result = RandomGenerator::get_instance().choice(evals, p_values->size());

	delete[] evals;

	return result;
}

void BoltzmanExploration::update(int p_t)
{
	if (_interpolation != nullptr)
	{
		_T = _interpolation->interpolate(p_t);
		cout << _T << endl;
	}
}
