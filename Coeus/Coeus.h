#pragma once
#pragma warning( disable : 4251)

namespace Coeus
{
	static const char NAME[] = "Coeus";
	static const char VERSION[] = "2.0.0";
	static const int BUILD = 1;

	static const float PI = 3.141592653589793238463f;
	static const float sqrt2PI = 2.50662827463f;

	enum ACTIVATION {
		LINEAR,
		BINARY,
		SIGMOID,
		TANH,
		SOFTMAX,
		SOFTPLUS,
		RELU,
		EXP,
		GAUSS
	};

	enum GRADIENT_RULE
	{
		ADADELTA_RULE,
		ADAGRAD_RULE,
		ADAMAX_RULE,
		ADAM_RULE,
		AMSGRAD_RULE,
		BACKPROP_RULE,
		NADAM_RULE,
		RMSPROP_RULE
	};

	enum INIT {
		DEBUG = 0,
		UNIFORM = 1,
		LECUN_UNIFORM = 2,
		GLOROT_UNIFORM = 3,
		IDENTITY = 4,
		NORMAL = 5,
		EXPONENTIAL = 6,
		HE_UNIFORM = 7,
		LECUN_NORMAL = 8,
		GLOROT_NORMAL = 9,
		HE_NORMAL = 10
	};

	enum RECURRENT_MODE
	{
		NONE = 0,
		BPTT = 1,
		RTRL = 2
	};
}