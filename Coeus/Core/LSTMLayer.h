#pragma once
#include "BaseLayer.h"
#include "Coeus.h"
#include "Param.h"
#include "IActivationFunction.h"
#include "NeuronOperator.h"
#include "TensorInitializer.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayer : public BaseLayer
	{
	public:
		LSTMLayer(const string& p_id, int p_dim, ACTIVATION p_activation, TensorInitializer* p_initializer, int p_in_dim = 0);
		explicit LSTMLayer(json p_data);
		LSTMLayer(LSTMLayer &p_copy, bool p_clone);
		~LSTMLayer();
		LSTMLayer* copy(bool p_clone) override;

		void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers) override;
		void activate() override;

		void calc_derivative(map<string, Tensor*>& p_derivative) override;
		void calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map) override;

		void reset() override;
		void copy_params(BaseLayer* p_source) override;
		json get_json() const override;

	private:
		Tensor* get_dim_tensor() override;
		
		NeuronOperator* _cec;
		NeuronOperator* _ig;
		NeuronOperator* _fg;
		NeuronOperator* _og;

		Param*		_Wxc;
		Param*		_Wxfg;
		Param*		_Wxig;
		Param*		_Wxog;

		Tensor*		_context;
		Tensor*		_state;

		IActivationFunction* _activation_function;
		TensorInitializer *_initializer;
	};
}


