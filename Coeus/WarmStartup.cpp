#include "WarmStartup.h"
#include <cmath>
#include "Coeus.h"
using namespace Coeus;


WarmStartup::WarmStartup(const float p_alpha_min, const float p_alpha_max, const int p_T0, const int p_Tmult):
	_alpha_min(p_alpha_min), 
	_alpha_max(p_alpha_max), 
	_Tcur(0), 
	_Ti(p_T0), 
	_Tmult(p_Tmult)
{
}

WarmStartup::~WarmStartup()
= default;

float WarmStartup::get_alpha()
{
	if (_Tcur > _Ti)
	{
		_Ti *= _Tmult;
		_Tcur = 0;
	}

	const float phi = static_cast<float>(_Tcur) / _Ti * PI;
	const float alpha = _alpha_min + 0.5 * (_alpha_max - _alpha_min) * (1 + cos(phi));

	_Tcur++;

	return alpha;
}
