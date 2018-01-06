#pragma once
#include "NeuralGroup.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) BaseLayer
{
public:
	enum TYPE
	{
		SOM = 1,
		MSOM = 2
	};

	BaseLayer();
	virtual ~BaseLayer();

	virtual void activate(Tensor* p_input) = 0;

	Tensor* get_output() const { return _output_group->getOutput(); }
	TYPE	type() const { return _type; }

protected:
	TYPE		_type;
	NeuralGroup *_input_group;
	NeuralGroup *_output_group;
};

}

