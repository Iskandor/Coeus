#include "LearningRateModule.h"
#include <cmath>
#include "FLAB.h"

using namespace Coeus;

LearningRateModule::LearningRateModule()
{
}


LearningRateModule::~LearningRateModule()
{
}

void LearningRateModule::init(const double p_alpha_min, const double p_alpha_max, const int p_T0, const int p_Tmult)
{
	_alpha_min = p_alpha_min;
	_alpha_max = p_alpha_max;
	_Tcur = 0;
	_Ti = p_T0;
	_Tmult = p_Tmult;
}

double LearningRateModule::update()
{

	if (_Tcur > _Ti)
	{
		_Ti *= _Tmult;
		_Tcur = 0;
	}

	const double phi = static_cast<float>(_Tcur) / _Ti * PI;
	const double alpha = _alpha_min + 0.5 * (_alpha_max - _alpha_min) * (1 + cos(phi));

	_Tcur++;

	return alpha;
}
