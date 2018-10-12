#pragma once
#pragma warning( disable : 4251)
#include <string>

static const string VERSION = "0.0.1";

enum ACTIVATION {
	LINEAR,
	BINARY,
	SIGMOID,
	TANH,
	SOFTMAX,
	SOFTPLUS,
	RELU,
	EXPONENTIAL,
	GAUSS
};