#include "ExponentialActivation.h"
#include <cmath>
#include <cstring>

using namespace Coeus;

ExponentialActivation::ExponentialActivation(const int p_k): IActivationFunction(EXPONENTIAL), _k(p_k) {
}

ExponentialActivation::~ExponentialActivation()
{
}

Tensor ExponentialActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = exp(-_k * p_input[i]);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor ExponentialActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size() * p_input.size());
	memset(arr, 0, sizeof(double) * p_input.size() * p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i*p_input.size() + i] = -exp(-_k * p_input[i]);
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);

}

json ExponentialActivation::get_json()
{
	json result = IActivationFunction::get_json();

	result["k"] = _k;

	return result;
}
