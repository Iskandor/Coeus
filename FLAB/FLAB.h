#pragma once

const double PI = 3.141592653589793238463;
const double sqrt2PI = 2.50662827463;

static int kronecker_delta(const int i, const int j) {
	return i == j ? 1 : 0;
}

static int sign(const double x)
{
	int result = 0;

	if (x > 0) result = 1;
	if (x < 0) result = -1;
	return result;
}