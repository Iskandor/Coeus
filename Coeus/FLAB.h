#pragma once

namespace FLAB {

const float PI = 3.141592653589793238463f;
const float sqrt2PI = 2.50662827463f;

static int kronecker_delta(const int i, const int j) {
	return i == j ? 1 : 0;
}

static int sign(const float x)
{
	int result = 0;

	if (x > 0) result = 1;
	if (x < 0) result = -1;
	return result;
}

static float max(float a, float b)
{
	return a > b ? a : b;
}

}