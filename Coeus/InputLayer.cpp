#include "InputLayer.h"
#include "IDGen.h"

using namespace Coeus;

InputLayer::InputLayer(const string& p_id, const int p_input_dim) : BaseLayer(p_id)
{
	_type = INPUT;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(p_input_dim, LINEAR, false));
	_input_group = _output_group = _group;
}

InputLayer::InputLayer(json p_data) : BaseLayer(p_data)
{
	_type = INPUT;
	_group = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["group"]));
	_input_group = _output_group = _group;
}

InputLayer::InputLayer(InputLayer& p_copy) : BaseLayer(IDGen::instance().next()) {
	_group = add_group<SimpleCellGroup>(p_copy._group->clone());
	_input_group = _output_group = _group;
	_type = INPUT;
}

InputLayer::~InputLayer()
{
	delete _group;
}

void InputLayer::integrate(Tensor* p_input, Tensor* p_weights) {
}

void InputLayer::activate(Tensor * p_input)
{
	if (p_input != nullptr) {
		_group->set_output(p_input);
	}	
}

void InputLayer::override(BaseLayer * p_source)
{
}

json InputLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["group"] = _group->get_json();

	return data;
}
