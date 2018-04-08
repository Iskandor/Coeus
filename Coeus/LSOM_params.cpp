#include "LSOM_params.h"

using namespace Coeus;

LSOM_params::LSOM_params(LSOM* p_lsom) : Base_SOM_params(p_lsom)
{
	_alpha = _alpha0 = 0;
	_beta = _beta0 = 0;
}

LSOM_params::~LSOM_params()
{
}

void LSOM_params::init_training(const double p_alpha, const double p_beta, const double p_epochs) {
	init(p_epochs);
	_alpha0 = p_alpha;
	_beta0 = p_beta;
	_alpha = _alpha0 * exp(-_iteration / _epochs);
	_beta = _beta0 * exp(-_iteration / _epochs);
}

void LSOM_params::param_decay() {
	Base_SOM_params::param_decay();
	_alpha = _alpha0 * exp(-_iteration / _epochs);
	_beta = _beta0 * exp(-_iteration / _epochs);
}
