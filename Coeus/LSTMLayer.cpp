#include "LSTMLayer.h"
#include "LSTMLayerGradient.h"

using namespace Coeus;


LSTMLayer::LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation) : BaseLayer(p_id)
{
	_type = LSTM;

	//_memory_cells = new LSTMCellGroup(p_dim, p_activation);

	_gradient_component = new LSTMLayerGradient(this);
}

LSTMLayer::~LSTMLayer()
{

}

void LSTMLayer::init(vector<BaseLayer*>& p_input_layers)
{

}

void LSTMLayer::integrate(Tensor* p_input, Tensor* p_weights)
{

}

void LSTMLayer::activate(Tensor* p_input)
{

}

void LSTMLayer::override(BaseLayer* p_source)
{
//#TODO doplnit prepis parametrov LSTM siete
}