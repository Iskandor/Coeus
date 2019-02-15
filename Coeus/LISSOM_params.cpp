#include "LISSOM_params.h"

using namespace Coeus;


LISSOM_params::LISSOM_params(LISSOM* p_lissom) : Base_SOM_params(p_lissom) {
	_alpha_a = 0;
	_alpha_e = 0;
	_alpha_i = 0;
}

LISSOM_params::~LISSOM_params()
{
}

void LISSOM_params::init_training(const float p_alpha_a, const float p_alpha_e, const float p_alpha_i) {
	_alpha_a = p_alpha_a;
	_alpha_e = p_alpha_e;
	_alpha_i = p_alpha_i;
}
