#pragma once
//#include <vld.h>

const double PI = 3.141592653589793238463;
const double sqrt2PI = 2.50662827463;

static int kronecker_delta(const int i, const int j) {
	return i == j ? 1 : 0;
}