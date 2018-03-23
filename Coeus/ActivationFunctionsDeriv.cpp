#include "ActivationFunctionsDeriv.h"
#include <cmath>

using namespace Coeus;

double ActivationFunctionsDeriv::dlinear(double p_x)
{
	return 1.0;
}

double Coeus::ActivationFunctionsDeriv::dbinary(double p_x)
{
	return 1.0;
}

double Coeus::ActivationFunctionsDeriv::dsigmoid(double p_x)
{
	return p_x * (1 - p_x);
}

double Coeus::ActivationFunctionsDeriv::dtanh(double p_x)
{
	return 1 - pow(p_x, 2);
}

double Coeus::ActivationFunctionsDeriv::dexponential(double p_x)
{
	return -exp(-p_x);
}

double Coeus::ActivationFunctionsDeriv::dsoftplus(double p_x)
{
	return 1 / (1 + exp(-p_x));
}

double Coeus::ActivationFunctionsDeriv::drelu(double p_x)
{
	return p_x > 0 ? 1 : 0;
}

double Coeus::ActivationFunctionsDeriv::dkexponential(double p_x)
{
	return NAN; // doplnit
}

double Coeus::ActivationFunctionsDeriv::dgauss(double p_x)
{
	return NAN; // doplnit
}
