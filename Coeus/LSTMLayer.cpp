#include "LSTMLayer.h"
#include "LSTMLayerGradient.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = LSTM;

	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, true));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, true));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, SIGMOID, true));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_dim, p_activation, _input_gate, _output_gate, _forget_gate));
	_context = add_group<SimpleCellGroup>(new SimpleCellGroup(p_dim, LINEAR, false));

	_aux_input = nullptr;
	_in_input_gate = nullptr;
	_in_output_gate = nullptr;
	_in_forget_gate = nullptr;

	_output_group = _cec;

	_ct_cec = add_connection(new Connection(p_dim, p_dim, _context->get_id(), _cec->get_id(), Connection::LECUN_UNIFORM));
}

LSTMLayer::LSTMLayer(json p_data) : BaseLayer(p_data)
{
	_type = LSTM;

	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["input_gate"]));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["output_gate"]));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["forget_gate"]));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_data["cec"], _input_gate, _output_gate, _forget_gate));
	_context = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["context"]));

	_output_group = _cec;

	_ct_cec = add_connection(new Connection(p_data["context_cec"]));

	_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(p_data["aux_input"]));
	_in_input_gate = add_connection(new Connection(p_data["in_input_gate"]));;
	_in_output_gate = add_connection(new Connection(p_data["in_output_gate"]));;
	_in_forget_gate = add_connection(new Connection(p_data["in_forget_gate"]));;
}

LSTMLayer::~LSTMLayer()
{
	delete _cec;
	delete _input_gate;
	delete _output_gate;
	delete _forget_gate;
	delete _aux_input;
	delete _input_group;
	delete _context;

	delete _in_input_gate;
	delete _in_output_gate;
	delete _in_forget_gate;
	delete _ct_cec;
}

LSTMLayer* LSTMLayer::clone()
{
	return new LSTMLayer(this);
}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers)
{
	int dim = 0;
	for (auto& layer : p_input_layers)
	{
		dim += layer->get_output()->size();
	}

	const int aux_dim = dim + _cec->get_dim();

	_input_group = add_group<SimpleCellGroup>(new SimpleCellGroup(dim, LINEAR, false));

	if (_aux_input == nullptr)
	{
		_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(aux_dim, LINEAR, false));
	}

	if (_in_input_gate == nullptr)
	{
		_in_input_gate = add_connection(new Connection(_aux_input->get_dim(), _input_gate->get_dim(), _aux_input->get_id(), _input_gate->get_id(), Connection::LECUN_UNIFORM));
		add_param(_in_input_gate);
	}

	if (_in_output_gate == nullptr)
	{
		_in_output_gate = add_connection(new Connection(_aux_input->get_dim(), _output_gate->get_dim(), _aux_input->get_id(), _output_gate->get_id(), Connection::LECUN_UNIFORM));
		add_param(_in_output_gate);
	}

	if (_in_forget_gate == nullptr)
	{
		_in_forget_gate = add_connection(new Connection(_aux_input->get_dim(), _forget_gate->get_dim(), _aux_input->get_id(), _forget_gate->get_id(), Connection::LECUN_UNIFORM));
		add_param(_in_forget_gate);
	}


}

void LSTMLayer::integrate(Tensor* p_input, Tensor* p_weights)
{
	_cec->integrate(p_input, p_weights);
	_input.emplace_back(p_input);
}

void LSTMLayer::activate(Tensor* p_input)
{
	_input_group->set_output(_input);
	_context->set_output(_cec->get_output());

	_input.emplace_back(_cec->get_output());
	_aux_input->set_output(_input);

	_input_gate->integrate(_aux_input->get_output(), _in_input_gate->get_weights());
	_input_gate->activate();
	_output_gate->integrate(_aux_input->get_output(), _in_output_gate->get_weights());
	_output_gate->activate();
	_forget_gate->integrate(_aux_input->get_output(), _in_forget_gate->get_weights());
	_forget_gate->activate();

	_cec->integrate(_context->get_output(), _ct_cec->get_weights());
	_cec->activate();

	_input.clear();
}

void LSTMLayer::override(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}

void LSTMLayer::reset()
{
	_cec->reset();
	_context->get_output()->fill(0);
}

json LSTMLayer::get_json() const
{
	json data = BaseLayer::get_json();

	data["input_gate"] = _input_gate->get_json();
	data["output_gate"] = _output_gate->get_json();
	data["forget_gate"] = _forget_gate->get_json();
	data["cec"] = _cec->get_json();
	data["context"] = _context->get_json();
	data["context_cec"] = _ct_cec->get_json();
	data["aux_input"] = _aux_input->get_json();
	data["in_input_gate"] = _in_input_gate->get_json();
	data["in_output_gate"] = _in_output_gate->get_json();
	data["in_forget_gate"] = _in_forget_gate->get_json();

	return data;
}

LSTMLayer::LSTMLayer(LSTMLayer* p_source) : BaseLayer(p_source)
{
	_type = LSTM;

	_input_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_input_gate));
	_output_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_output_gate));
	_forget_gate = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_forget_gate));
	_cec = add_group<LSTMCellGroup>(new LSTMCellGroup(p_source->_cec, _input_gate, _output_gate, _forget_gate));
	_context = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_context));

	_aux_input = add_group<SimpleCellGroup>(new SimpleCellGroup(p_source->_aux_input));
	_in_input_gate = add_connection(p_source->_in_input_gate->clone());
	_in_output_gate = add_connection(p_source->_in_output_gate->clone());
	_in_forget_gate = add_connection(p_source->_in_forget_gate->clone());

	_output_group = _cec;

	_ct_cec = add_connection(p_source->_ct_cec->clone());
}
