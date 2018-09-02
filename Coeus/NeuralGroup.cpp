#include "NeuralGroup.h"
#include "ActivationFunctions.h"
#include "IDGen.h"

using namespace std;
using namespace Coeus;
/**
 * NeuralGroup constructor creates layer of p_dim neurons with p_activationFunction
 * @param p_dim dimension of layer
 * @param p_activation_function get_type of activation function
 * @param p_bias
 */
NeuralGroup::NeuralGroup(const int p_dim, const ACTIVATION p_activation_function, const bool p_bias)
{
    _id = IDGen::instance().next();
	_bias_flag = p_bias;	
	
    _dim = p_dim;
    activation_function_ = p_activation_function;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
	_bias = Tensor::Random({ _dim}, 1);
}

NeuralGroup::NeuralGroup(nlohmann::json p_data) {
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	activation_function_ = static_cast<ACTIVATION>(p_data["actfn"].get<int>());
	_bias_flag = p_data["bias"].get<bool>();
	//#TODO doriesit _bias = ;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(NeuralGroup &p_copy) {
    _id = p_copy._id;
    _dim = p_copy._dim;
    activation_function_ = p_copy.activation_function_;
	_bias = Tensor(p_copy._bias);

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
    _bias_flag = p_copy._bias_flag;
}

/**
 * NeuralGroup destructor
 */
NeuralGroup::~NeuralGroup(void)
{

}

/**
 * performs product of weights and input which is stored in actionPotential vector
 * @param p_input vector of input values
 * @param p_weights matrix of input connection params
 */
void NeuralGroup::integrate(Tensor* p_input, Tensor* p_weights) {
    _ap += (*p_weights) * (*p_input);
}

/**
 * calculates the get_output of layer according to activation function
 */
void NeuralGroup::activate() {
	if (is_bias()) {
		_ap += _bias;
	}

    switch (activation_function_) {
        case LINEAR:
			_output = Tensor::apply(_ap, ActivationFunctions::linear);
            break;
        case BINARY:
			_output = Tensor::apply(_ap, ActivationFunctions::binary);
            break;
        case SIGMOID:
			_output = Tensor::apply(_ap, ActivationFunctions::sigmoid);
            break;
        case TANH:
			_output = Tensor::apply(_ap, ActivationFunctions::tanh);
            break;
        case SOFTPLUS:
			_output = Tensor::apply(_ap, ActivationFunctions::softplus);
            break;
        case RELU:
			_output = Tensor::apply(_ap, ActivationFunctions::relu);
            break;
	    default: ;
    }
	_ap.fill(0);
}

void NeuralGroup::set_output(Tensor* p_output) const {
    _output.override(p_output);
}

void NeuralGroup::update_bias(Tensor& p_delta_b) {
	_bias += p_delta_b;
}

