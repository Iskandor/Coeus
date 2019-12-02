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
		RMSPROP_RULE,
		RADAM_RULE
	};

	enum RECURRENT_MODE
	{
		NONE = 0,
		BPTT = 1,
		RTRL = 2
	};
}