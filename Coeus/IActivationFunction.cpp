#include "IActivationFunction.h"

using namespace Coeus;

IActivationFunction::IActivationFunction(const ACTIVATION p_id): _type(p_id) {
}

IActivationFunction::~IActivationFunction()
= default;

json IActivationFunction::get_json()
{
	json data;

	data["type"] = _type;

	return data;
}
