#include "Base_SOM_params.h"
#include <cmath>

using namespace Coeus;

Base_SOM_params::Base_SOM_params(SOM* p_som): _sigma(0), _iteration(0) {
	_sigma0 = sqrt(pow(max(p_som->dim_x(), p_som->dim_y()), 2) * 2);
	_lambda = 1;
}


Base_SOM_params::~Base_SOM_params()
{
}

void Base_SOM_params::init(const double p_epochs)
{
	_iteration = 0;
	_epochs = p_epochs;
	_lambda = p_epochs / log(_sigma0);
	_sigma = _sigma0 * exp(-_iteration / _lambda);
}

void Base_SOM_params::param_decay()
{
	_iteration++;
	_sigma = _sigma0 * exp(-_iteration / _lambda);
}
