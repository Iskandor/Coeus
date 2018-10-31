#pragma once
#include "SimpleCellGroup.h"
#include "Connection.h"
#include "ParamModel.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) BaseLayer : public ParamModel
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
	BaseLayer(json p_data);
	virtual ~BaseLayer();
	virtual BaseLayer* clone() = 0;

	virtual void init(vector<BaseLayer*>& p_input_layers);
	virtual void integrate(Tensor* p_input, Tensor* p_weights) = 0;
	virtual void activate(Tensor* p_input = nullptr) = 0;
	virtual void override(BaseLayer* p_source) = 0;
	virtual void reset() = 0;

	TYPE	get_type() const { return _type; }
	string	get_id() const { return _id; }

	int input_dim() const { return _input_group->get_dim(); }
	int output_dim() const { return _output_group->get_dim(); }

	bool is_valid() const { return _valid; }
	void set_valid(const bool p_val) { _valid = p_val; }

	template<typename T>
	T* get_output_group() const { return static_cast<T*>(_output_group); }
	template<typename T>
	T* get_input_group() const { return static_cast<T*>(_input_group); }

	map<string, Connection*>* get_connections() { return &_connections; }
	map<string, BaseCellGroup*>* get_groups() { return &_groups; }

	Tensor* get_output() const { return _output_group->get_output(); }

	virtual json get_json() const;

protected:
	explicit BaseLayer(BaseLayer* p_source);
	Connection*		add_connection(Connection* p_connection);
	template<typename T>
	T*	add_group(T* p_group);

	string		_id;
	TYPE		_type;

	map<string, Connection*>	_connections;
	map<string, BaseCellGroup*>	_groups;	

	BaseCellGroup* _output_group;
	BaseCellGroup* _input_group;

private:
	bool	_valid;
};

template <typename T>
T* BaseLayer::add_group(T* p_group)
{
	_groups[p_group->get_id()] = p_group;
	add_param(p_group);

	return p_group;
}
}