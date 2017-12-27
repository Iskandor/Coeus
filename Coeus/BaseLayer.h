#pragma once
#include "NeuralGroup.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) BaseLayer
{
public:
	BaseLayer();
	virtual ~BaseLayer();

	virtual void activate(Tensor* p_input) = 0;

	Tensor* get_output() { return _output_group->getOutput(); }

protected:
	NeuralGroup *_input_group;
	NeuralGroup *_output_group;
};

}

