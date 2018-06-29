#include "LSOM1_params.h"

using namespace Coeus;

LSOM1_params::LSOM1_params(LSOM1* p_lsom) : Base_SOM_params(p_lsom)
{
	_alpha = _alpha0 = 0;
	_beta = _beta0 = 0;
}

LSOM1_params::~LSOM1_params()
{
}

void LSOM1_params::init_training(const double p_alpha, const double p_beta, const double p_epochs) {
	init(p_epochs);
	_alpha0 = p_alpha;
	_beta0 = p_beta;
	_alpha = _alpha0 * exp(-_iteration / _epochs);
	_beta = _beta0; // / exp(-_iteration / _epochs);
}

void LSOM1_params::param_decay() {
	Base_SOM_params::param_decay();
	_alpha = _alpha0 * exp(-_iteration / _epochs);
	//_beta = _beta0  / exp(-_iteration / _epochs);
}
