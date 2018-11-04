#include "WarmStartup.h"
#include <cmath>
#include "FLAB.h"

using namespace Coeus;

WarmStartup::WarmStartup(const double p_alpha_min, const double p_alpha_max, const int p_T0, const int p_Tmult):
	_alpha_min(p_alpha_min), 
	_alpha_max(p_alpha_max), 
	_Tcur(0), 
	_Ti(p_T0), 
	_Tmult(p_Tmult)
{
}

WarmStartup::~WarmStartup()
= default;

double WarmStartup::get_alpha()
{
	if (_Tcur > _Ti)
	{
		_Ti *= _Tmult;
		_Tcur = 0;
	}

	const double phi = static_cast<double>(_Tcur) / _Ti * PI;
	const double alpha = _alpha_min + 0.5 * (_alpha_max - _alpha_min) * (1 + cos(phi));

	_Tcur++;

	return alpha;
}
