#pragma once
#include "NeuralGroup.h"
#include "Connection.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) BaseLayer
{
	friend class IGradientComponent;
public:
	enum TYPE
	{
		SOM = 1,
		MSOM = 2,
		INPUT = 3,
		CORE = 4,
		RECURRENT = 5,
		LSTM = 6,
		LSOM = 7
	};

	BaseLayer(string p_id);
	BaseLayer(nlohmann::json p_data);
	virtual ~BaseLayer();

	virtual void integrate(Tensor* p_input, Tensor* p_weights) = 0;
	virtual void activate(Tensor* p_input = nullptr) = 0;
	virtual void override_params(BaseLayer* p_source) = 0;
	virtual void post_connection(BaseLayer* p_input) = 0;

	Tensor* get_output() const { return _output_group->get_output(); }
	TYPE	type() const { return _type; }
	string	id() const { return _id; }

	int input_dim() const { return _input_group->get_dim(); }
	int output_dim() const { return _output_group->get_dim(); }

	bool is_valid() const { return _valid; }
	void set_valid(const bool p_val) { _valid = p_val; }

	IGradientComponent* gradient_component() const { return _gradient_component; }

	NeuralGroup* get_output_group() const { return _output_group; }
	NeuralGroup* get_input_group() const { return _input_group; }

protected:
	Connection* add_connection(Connection* p_connection);

	string		_id;
	TYPE		_type;
	NeuralGroup *_input_group;
	NeuralGroup *_output_group;

	IGradientComponent* _gradient_component;

	map<string, Connection*> _connections;

private:
	bool	_valid;
};

}

