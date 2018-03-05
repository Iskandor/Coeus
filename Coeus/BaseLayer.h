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
		MSOM = 2,
		INPUT = 3,
		CORE = 4,
		RECURRENT = 5,
		LSTM = 6
	};

	BaseLayer(string p_id);
	BaseLayer(nlohmann::json p_data);
	virtual ~BaseLayer();

	virtual void activate(Tensor* p_input) = 0;
	virtual void override_params(BaseLayer* p_source) = 0;

	Tensor* get_output() const { return _output_group->getOutput(); }
	TYPE	type() const { return _type; }
	string	id() const { return _id; }

	int input_dim() const { return _input_group->getDim(); }
	int output_dim() const { return _output_group->getDim(); }

protected:
	string		_id;
	TYPE		_type;
	NeuralGroup *_input_group;
	NeuralGroup *_output_group;
};

}

