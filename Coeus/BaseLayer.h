#pragma once
#include <string>
#include <map>
#include "NeuralGroup.h"
#include "Connection.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) BaseLayer
{
public:
	BaseLayer();
	virtual ~BaseLayer();

	virtual void activate(Tensor* p_input) = 0;

protected:
	NeuralGroup* add_group(int p_dim, NeuralGroup::ACTIVATION p_activation, bool p_bias);
	Connection* add_connection(NeuralGroup* p_in_group, NeuralGroup* p_out_group, Connection::INIT p_init, double p_limit);

	Connection* get_connection(string p_input_group, string p_output_group);

	map<string, NeuralGroup*> _groups;
	map<string, Connection*> _connections;
	map<string, vector<string>> _graph;

	string _inputGroup;
	string _outputGroup;
};

}

