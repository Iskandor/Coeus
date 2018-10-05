#pragma once
#include "SimpleCellGroup.h"
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

	BaseLayer(const string& p_id);
	BaseLayer(nlohmann::json p_data);
	virtual ~BaseLayer();

	virtual void init(vector<BaseLayer*>& p_input_layers);
	virtual void integrate(Tensor* p_input, Tensor* p_weights) = 0;
	virtual void activate(Tensor* p_input = nullptr) = 0;
	virtual void override(BaseLayer* p_source) = 0;
	void update(map<string, Tensor> &p_update);

	TYPE	get_type() const { return _type; }
	string	id() const { return _id; }

	int input_dim() const { return _input_group->get_dim(); }
	int output_dim() const { return _output_group->get_dim(); }

	bool is_valid() const { return _valid; }
	void set_valid(const bool p_val) { _valid = p_val; }

	IGradientComponent* gradient_component() const { return _gradient_component; }

	BaseCellGroup* get_output_group() const { return _output_group; }
	BaseCellGroup* get_input_group() const { return _input_group; }

	map<string, Connection*>* get_connections() { return &_connections; }
	map<string, BaseCellGroup*>* get_groups() { return &_groups; }

	Tensor* get_output() const { return _output_group->get_output(); }

protected:
	Connection*		add_connection(Connection* p_connection);
	template<typename T>
	T*	add_group(T* p_group);

	string		_id;
	TYPE		_type;

	IGradientComponent* _gradient_component;

	map<string, Connection*>	_connections;
	map<string, BaseCellGroup*>	_groups;
	map<string, Tensor*>		_params;

	BaseCellGroup* _output_group;
	BaseCellGroup* _input_group;

private:
	bool	_valid;
};

template <typename T>
T* BaseLayer::add_group(T* p_group)
{
	_groups[p_group->get_id()] = p_group;

	if (dynamic_cast<SimpleCellGroup*>(p_group) != nullptr)
	{
		SimpleCellGroup* group = dynamic_cast<SimpleCellGroup*>(p_group);
		if (group->is_bias())
		{
			_params[p_group->get_id()] = group->get_bias();
		}		
	}

	return p_group;
}
}

