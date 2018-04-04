#include "SOM_params.h"

using namespace Coeus;

SOM_params::SOM_params(SOM* p_som): Base_SOM_params(p_som) {
	_alpha = _alpha0 = 0;
}


SOM_params::~SOM_params()
{
}

void SOM_params::init_training(const double p_alpha, const double p_epochs) {
	init(p_epochs);
	_alpha0 = p_alpha;
	_alpha = _alpha0 * exp(-_iteration / _epochs);
}

void SOM_params::param_decay() {
	Base_SOM_params::param_decay();
	_alpha = _alpha0 * exp(-_iteration / _epochs);
}
