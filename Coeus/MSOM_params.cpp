#include "MSOM_params.h"

using namespace Coeus;

MSOM_params::MSOM_params(MSOM* p_som): Base_SOM_params(p_som) {
	_gamma1 = _gamma1_0 = 0;
	_gamma2 = _gamma2_0 = 0;
}


MSOM_params::~MSOM_params()
{
}

void MSOM_params::init_training(const double p_gamma1, const double p_gamma2, const double p_epochs) {
	init(p_epochs);
	_gamma1_0 = p_gamma1;
	_gamma2_0 = p_gamma2;
	_gamma1 = _gamma1_0 * exp(-_iteration / _epochs);
	_gamma2 = _gamma2_0 * exp(-_iteration / _epochs);
}

void MSOM_params::param_decay() {
	Base_SOM_params::param_decay();
	_gamma1 = _gamma1_0 * exp(-_iteration / _epochs);
	_gamma2 = _gamma2_0 * exp(-_iteration / _epochs);
}
